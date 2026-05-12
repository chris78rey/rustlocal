// ============================================================================
// perfilador-parquet — Perfilador de archivos Parquet con egui + Polars
// Lee un .parquet, genera perfil de columnas, calcula correlaciones Pearson,
// agrupa columnas correlacionadas (mín. 3), muestra en GUI y exporta CSV.
// Incluye barra de progreso real con hilos.
// Ejecutar: cargo run --bin perfilador-parquet
// ============================================================================

use anyhow::{Context, Result};
use eframe::egui;
use egui::Panel;
use polars::prelude::*;
use rfd::FileDialog;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// ─── Estados del procesamiento ──────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
enum ProcEstado { Idle, Processing, Done, Error(String) }

#[derive(Clone)]
struct ProcState {
    estado: ProcEstado,
    progress: f32,
    mensaje: String,
    resultado: Option<ResultadoPerfil>,
}
impl Default for ProcState {
    fn default() -> Self { Self { estado: ProcEstado::Idle, progress: 0.0, mensaje: "Esperando archivo...".into(), resultado: None } }
}

#[derive(Default, Clone)]
struct ResultadoPerfil {
    filas: usize,
    columnas: usize,
    carpeta_salida: PathBuf,
    perfil: Vec<PerfilColumna>,
    correlaciones: Vec<Correlacion>,
    grupos: Vec<GrupoCorrelacion>,
}

#[derive(Clone)]
struct PerfilColumna {
    columna: String,
    tipo_polars: String,
    total_filas: usize,
    no_nulos: usize,
    nulos: usize,
    porcentaje_nulos: f64,
    valores_unicos: Option<usize>,
    media: Option<f64>,
    mediana: Option<f64>,
    minimo: Option<f64>,
    maximo: Option<f64>,
    desviacion_estandar: Option<f64>,
}

#[derive(Clone)]
struct Correlacion {
    columna_1: String,
    columna_2: String,
    correlacion: f64,
    correlacion_absoluta: f64,
    fuerza: String,
}

#[derive(Clone)]
struct GrupoCorrelacion {
    columnas: Vec<String>,
    tamano: usize,
    correlacion_minima: f64,
    correlacion_promedio: f64,
}

// ─── Constantes visuales ────────────────────────────────────────────────────

const BORDE_GRUESO: f32 = 2.0;
const ESQUINA: f32 = 8.0;

/// Paleta amarillo ↔ agua (alta visibilidad)
// Amarillo
const A1: egui::Color32 = egui::Color32::from_rgb(245, 210, 50);   // amarillo sol
const A2: egui::Color32 = egui::Color32::from_rgb(250, 235, 140); // amarillo claro
const A3: egui::Color32 = egui::Color32::from_rgb(200, 160, 30);  // amarillo oscuro
const A4: egui::Color32 = egui::Color32::from_rgb(255, 248, 210); // amarillo fondo
// Agua
const W1: egui::Color32 = egui::Color32::from_rgb(50, 185, 195);  // agua medio
const W2: egui::Color32 = egui::Color32::from_rgb(120, 220, 230); // agua claro
const W3: egui::Color32 = egui::Color32::from_rgb(25, 140, 155);  // agua profundo
const W4: egui::Color32 = egui::Color32::from_rgb(210, 245, 248); // agua fondo

const FONDO_TARJETA: egui::Color32 = A4;   // fondo amarillo suave
const FONDO_PANEL: egui::Color32 = A4;     // fondo paneles
const FONDO_GLOBAL: egui::Color32 = W4;    // fondo general agua claro
const BORDE: egui::Color32 = egui::Color32::from_rgb(60, 140, 150); // borde agua oscuro

/// Colores cíclicos para grupos (amarillo ↔ agua)
const COLORES_GRUPO: &[egui::Color32] = &[
    A1, W1, A3, W3, A2, W2,
];

// ─── Estado de la App ───────────────────────────────────────────────────────

struct PerfilApp {
    archivo: Option<PathBuf>,
    umbral_texto: String,
    min_grupo_texto: String,
    proc: Arc<Mutex<ProcState>>,
    tab_activa: usize,
}

const TAB_LABELS: &[&str] = &["📋 Resumen", "📑 Perfil", "🔗 Correlaciones", "📦 Grupos"];

impl eframe::App for PerfilApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Fondo global
        ui.painter().rect_filled(ui.max_rect(), 0.0, FONDO_GLOBAL);

        if self.umbral_texto.is_empty() { self.umbral_texto = "0.70".into(); }
        if self.min_grupo_texto.is_empty() { self.min_grupo_texto = "3".into(); }

        let ctx = ui.ctx().clone();
        let proc = self.proc.clone();

        // ═══════════ PANEL SUPERIOR: CONTROLES ═══════════════════════════════
        Panel::top("panel_controles").show_inside(ui, |pui| {
                pui.heading("📊 Perfilador de archivos Parquet");
                pui.separator();

                // Botones
                pui.horizontal(|hui| {
                    let ocupado = proc.lock().unwrap().estado == ProcEstado::Processing;
                    let archivo_ok = self.archivo.is_some();

                    if hui.add_enabled(!ocupado, egui::Button::new("📂 Seleccionar archivo"))
                        .on_hover_text("Selecciona un archivo .parquet")
                        .clicked()
                    {
                        if let Some(path) = FileDialog::new().add_filter("Parquet", &["parquet"]).pick_file() {
                            self.archivo = Some(path);
                            let mut p = proc.lock().unwrap();
                            p.estado = ProcEstado::Idle; p.progress = 0.0;
                            p.mensaje = "Archivo seleccionado.".into(); p.resultado = None;
                        }
                    }

                    if hui.add_enabled(archivo_ok && !ocupado, egui::Button::new("⚙️  Procesar"))
                        .on_hover_text("Inicia el perfilamiento y cálculo de correlaciones")
                        .clicked()
                    {
                        let path = self.archivo.clone().unwrap();
                        let umbral = self.umbral_texto.trim().parse::<f64>().unwrap_or(0.70);
                        let min_grupo = self.min_grupo_texto.trim().parse::<usize>().unwrap_or(3);
                        let pc = proc.clone();
                        { let mut p = proc.lock().unwrap();
                          p.estado = ProcEstado::Processing; p.progress = 0.0;
                          p.mensaje = "Iniciando...".into(); p.resultado = None; }
                        thread::spawn(move || {
                            match procesar_parquet(&path, umbral, min_grupo, pc.clone()) {
                                Ok(r) => { let mut p = pc.lock().unwrap();
                                    p.estado = ProcEstado::Done; p.progress = 1.0;
                                    p.mensaje = "✅ Finalizado correctamente.".into(); p.resultado = Some(r); }
                                Err(e) => { let mut p = pc.lock().unwrap();
                                    p.estado = ProcEstado::Error(format!("{e:#}"));
                                    p.mensaje = format!("❌ Error: {e:#}"); }
                            }
                        });
                    }
                });

                // Parámetros
                pui.horizontal(|hui| {
                    hui.label("🎯 Umbral:");
                    hui.add(egui::TextEdit::singleline(&mut self.umbral_texto).desired_width(50.0));
                    hui.separator();
                    hui.label("📦 Mín. columnas por grupo:");
                    hui.add(egui::TextEdit::singleline(&mut self.min_grupo_texto).desired_width(40.0));
                });

                // Archivo
                if let Some(path) = &self.archivo {
                    pui.label(format!("📁 {}", path.display()));
                } else {
                    pui.label("📁 Ningún archivo seleccionado");
                }

                // Barra de progreso con borde
                let p_state = proc.lock().unwrap().clone();
                if p_state.estado == ProcEstado::Processing || p_state.estado == ProcEstado::Idle {
                    let bar = egui::ProgressBar::new(p_state.progress)
                        .show_percentage()
                        .animate(p_state.estado == ProcEstado::Processing)
                        .fill(if p_state.estado == ProcEstado::Processing { W1 } else { egui::Color32::DARK_GRAY });
                    pui.add(bar);
                }
                pui.label(format!("📌 {}", p_state.mensaje));
                if p_state.estado == ProcEstado::Processing { ctx.request_repaint_after(Duration::from_millis(50)); }
            });

        // ═══════════ BARRA DE TABS (con borde) ═══════════════════════════════
        egui::Frame::new()
            .fill(FONDO_GLOBAL)
            .stroke(egui::Stroke::new(BORDE_GRUESO, BORDE))
            .corner_radius(ESQUINA)
            .show(ui, |fui| {
            Panel::top("panel_tabs").show_inside(fui, |tui| {
                tui.horizontal(|hui| {
                    for (i, label) in TAB_LABELS.iter().enumerate() {
                        let selected = self.tab_activa == i;
                        let mut btn = egui::Button::selectable(selected, *label);
                        if selected {
                            btn = btn.fill(W1);
                            btn = btn.stroke(egui::Stroke::new(2.0, A1));
                        } else {
                            btn = btn.stroke(egui::Stroke::new(1.0, BORDE));
                        }
                        if hui.add(btn).clicked() { self.tab_activa = i; }
                    }
                });
            }); });

        // ═══════════ PANEL CENTRAL (con borde) ═══════════════════════════════
        egui::Frame::new()
            .fill(FONDO_GLOBAL)
            .stroke(egui::Stroke::new(BORDE_GRUESO, BORDE))
            .corner_radius(ESQUINA)
            .show(ui, |cui| {
                egui::CentralPanel::default().show_inside(cui, |cui2| {
                    egui::ScrollArea::vertical().show(cui2, |sui| {
                        let resultado = proc.lock().unwrap().resultado.clone();
                        match self.tab_activa {
                            0 => tab_resumen(sui, &resultado),
                            1 => tab_perfil(sui, &resultado),
                            2 => tab_correlaciones(sui, &resultado),
                            3 => tab_grupos(sui, &resultado, &self.min_grupo_texto),
                            _ => {}
                        }
                    });
                });
            });
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Tabs
// ═════════════════════════════════════════════════════════════════════════════

fn marco_tab(ui: &mut egui::Ui, titulo: &str, contenido: impl FnOnce(&mut egui::Ui)) {
    ui.heading(titulo);
    ui.separator();
    egui::Frame::new()
        .stroke(egui::Stroke::new(1.5, BORDE))
        .corner_radius(6.0)
        .fill(FONDO_TARJETA)
        .show(ui, |ui| {
            ui.add_space(8.0);
            contenido(ui);
            ui.add_space(8.0);
        });
}

fn tab_resumen(ui: &mut egui::Ui, resultado: &Option<ResultadoPerfil>) {
    marco_tab(ui, "📋 Resumen General", |ui| {
        match resultado {
            None => {
                ui.add_space(60.0);
                ui.vertical_centered(|ui| {
                    ui.label("📭");
                    ui.add_space(12.0);
                    ui.heading("Selecciona un archivo Parquet y presiona Procesar.");
                });
            }
            Some(r) => {
                egui::Grid::new("resumen_grid")
                    .striped(true)
                    .min_col_width(220.0)
                    .show(ui, |ui| {
                        ui.strong("Métrica"); ui.strong("Valor"); ui.end_row();
                        ui.label("Filas");            ui.label(r.filas.to_string());            ui.end_row();
                        ui.label("Columnas");         ui.label(r.columnas.to_string());         ui.end_row();
                        ui.label("Carpeta de salida"); ui.label(r.carpeta_salida.display().to_string()); ui.end_row();
                        ui.label("Pares correlacionados"); ui.label(r.correlaciones.len().to_string()); ui.end_row();
                        ui.label("Grupos encontrados"); ui.label(r.grupos.len().to_string());   ui.end_row();
                    });

                ui.add_space(16.0);
                ui.separator();
                ui.label("📄 Archivos generados:");
                for f in ["perfil_general.csv", "correlaciones_fuertes.csv", "grupos_correlacion.csv"] {
                    ui.label(format!("   • {f}"));
                }
            }
        }
    });
}

fn tab_perfil(ui: &mut egui::Ui, resultado: &Option<ResultadoPerfil>) {
    marco_tab(ui, "📑 Perfil de Columnas", |ui| {
        match resultado {
            None => { ui.add_space(60.0); ui.vertical_centered(|ui| { ui.heading("Procesa un archivo para ver el perfil."); }); }
            Some(r) => {
                ui.label(format!("Mostrando primeras 200 de {} columnas.", r.perfil.len()));
                ui.add_space(8.0);
                egui::Grid::new("tabla_perfil").striped(true).min_col_width(80.0).show(ui, |ui| {
                    ui.strong("Columna"); ui.strong("Tipo"); ui.strong("Nulos"); ui.strong("% Nulos");
                    ui.strong("Únicos"); ui.strong("Media"); ui.strong("Mín"); ui.strong("Máx"); ui.strong("Desv.Est.");
                    ui.end_row();
                    for p in r.perfil.iter().take(200) {
                        ui.label(&p.columna); ui.label(&p.tipo_polars); ui.label(p.nulos.to_string());
                        ui.label(format!("{:.2}", p.porcentaje_nulos)); ui.label(opt_usize(p.valores_unicos));
                        ui.label(opt_f64(p.media)); ui.label(opt_f64(p.minimo));
                        ui.label(opt_f64(p.maximo)); ui.label(opt_f64(p.desviacion_estandar));
                        ui.end_row();
                    }
                });
            }
        }
    });
}

fn tab_correlaciones(ui: &mut egui::Ui, resultado: &Option<ResultadoPerfil>) {
    marco_tab(ui, "🔗 Correlaciones Fuertes", |ui| {
        match resultado {
            None => { ui.add_space(60.0); ui.vertical_centered(|ui| { ui.heading("Procesa un archivo para ver correlaciones."); }); }
            Some(r) => {
                if r.correlaciones.is_empty() {
                    ui.label("No se encontraron correlaciones fuertes con el umbral actual."); return;
                }
                ui.label(format!("Mostrando primeras 200 de {} pares.", r.correlaciones.len()));
                ui.add_space(8.0);
                egui::Grid::new("tabla_correlaciones").striped(true).min_col_width(100.0).show(ui, |ui| {
                    ui.strong("#"); ui.strong("Columna 1"); ui.strong("Columna 2"); ui.strong("Correlación"); ui.strong("Fuerza");
                    ui.end_row();
                    for (i, c) in r.correlaciones.iter().take(200).enumerate() {
                        ui.label(format!("{}", i + 1)); ui.label(&c.columna_1); ui.label(&c.columna_2);
                        let color = if c.correlacion > 0.0 { W1 } else { A1 };
                        ui.colored_label(color, format!("{:.6}", c.correlacion));
                        ui.label(&c.fuerza);
                        ui.end_row();
                    }
                });
            }
        }
    });
}

fn tab_grupos(ui: &mut egui::Ui, resultado: &Option<ResultadoPerfil>, min_str: &str) {
    marco_tab(ui, "📦 Grupos de Columnas Correlacionadas", |ui| {
        match resultado {
            None => { ui.add_space(60.0); ui.vertical_centered(|ui| { ui.heading("Procesa un archivo para ver grupos."); }); }
            Some(r) => {
                if r.grupos.is_empty() {
                    ui.label(format!("No se encontraron grupos con ≥ {} columnas.", min_str)); return;
                }
                ui.label(format!("Total: {} grupos (mín. {min_str} columnas)", r.grupos.len()));
                ui.add_space(12.0);

                for (idx, grupo) in r.grupos.iter().enumerate() {
                    let color = COLORES_GRUPO[idx % COLORES_GRUPO.len()];

                    egui::Frame::new()
                        .fill(FONDO_TARJETA)
                        .corner_radius(ESQUINA)
                        .stroke(egui::Stroke::new(BORDE_GRUESO, color))
                        .show(ui, |ui| {
                            ui.horizontal(|hui| {
                                // Barra indicadora lateral gruesa
                                let (rect, _) = hui.allocate_exact_size(
                                    egui::vec2(6.0, 56.0), egui::Sense::hover());
                                hui.painter().rect_filled(rect, 3.0, color);
                                hui.add_space(12.0);

                                hui.vertical(|vui| {
                                    let (rp, rm) = if grupo.tamano > 1 {
                                        (grupo.correlacion_promedio, grupo.correlacion_minima)
                                    } else { (1.0, 1.0) };

                                    vui.strong(format!(
                                        "Grupo #{} — {} columnas  |  r̅ = {:.4}  |  r_min = {:.4}",
                                        idx + 1, grupo.tamano, rp, rm,
                                    ));

                                    // Barra horizontal con borde
                                    let ancho = 220.0;
                                    let (rect, _) = vui.allocate_exact_size(
                                        egui::vec2(ancho, 16.0), egui::Sense::hover());
                                    let bg = egui::Rect::from_min_size(rect.min, egui::vec2(ancho, 16.0));
                                    vui.painter().rect_stroke(bg, 6.0, egui::Stroke::new(1.0, BORDE), egui::StrokeKind::Inside);
                                    vui.painter().rect_filled(bg, 6.0, egui::Color32::from_gray(35));
                                    let fill_w = (rp as f32).clamp(0.0, 1.0) * ancho;
                                    if fill_w > 0.0 {
                                        let fill = egui::Rect::from_min_size(rect.min, egui::vec2(fill_w, 16.0));
                                        let bar_color = if rp >= 0.9 { W1 }
                                            else if rp >= 0.7 { A1 }
                                            else { A2 };
                                        vui.painter().rect_filled(fill, 6.0, bar_color);
                                    }

                                    vui.add_space(6.0);

                                    // Columnas del grupo
                                    for col_name in &grupo.columnas {
                                        vui.label(format!("    • {col_name}"));
                                    }
                                });
                            });
                        });
                    ui.add_space(10.0);
                }
            }
        }
    });
}

// ═════════════════════════════════════════════════════════════════════════════
//  Lógica de procesamiento (hilo separado)
// ═════════════════════════════════════════════════════════════════════════════

fn procesar_parquet(path: &Path, umbral: f64, min_grupo: usize, proc: Arc<Mutex<ProcState>>) -> Result<ResultadoPerfil> {
    set_progress(&proc, 0.02, "Abriendo archivo Parquet...")?;
    let mut file = File::open(path).with_context(|| format!("No se pudo abrir {}", path.display()))?;

    set_progress(&proc, 0.05, "Leyendo Parquet...")?;
    let df = ParquetReader::new(&mut file).finish().context("No se pudo leer Parquet")?;

    let filas = df.height();
    let columnas = df.width();
    set_progress(&proc, 0.10, &format!("Leído: {filas} filas x {columnas} columnas"))?;

    let carpeta_salida = crear_carpeta_salida(path)?;
    let mut perfil: Vec<PerfilColumna> = Vec::new();
    let mut cols_num: Vec<(String, Vec<Option<f64>>)> = Vec::new();
    let cols = df.columns();
    let total = cols.len();

    for (i, col) in cols.iter().enumerate() {
        let s: &Series = col.as_materialized_series();
        let nombre = s.name().to_string();
        let tipo = format!("{:?}", s.dtype());
        let t = s.len();
        let nulos = s.null_count();
        let no_nulos = t.saturating_sub(nulos);
        let pct_n = if t > 0 { (nulos as f64 / t as f64) * 100.0 } else { 0.0 };
        let unicos = s.n_unique().ok();
        let mut media = None; let mut mediana = None; let mut minimo = None;
        let mut maximo = None; let mut desv = None;

        if es_num(s.dtype()) {
            if let Ok(f64) = s.cast(&DataType::Float64) {
                media = f64.mean(); mediana = f64.median(); desv = f64.std(1);
                minimo = f64.min::<f64>().ok().flatten(); maximo = f64.max::<f64>().ok().flatten();
                if let Ok(ca) = f64.f64() {
                    cols_num.push((nombre.clone(), ca.into_iter().collect()));
                }
            }
        }

        perfil.push(PerfilColumna { columna: nombre, tipo_polars: tipo, total_filas: t, no_nulos, nulos,
            porcentaje_nulos: pct_n, valores_unicos: unicos, media, mediana, minimo, maximo, desviacion_estandar: desv });

        let pct = 0.10 + (i as f64 / total as f64) * 0.40;
        set_progress(&proc, pct as f32, &format!("Perfilando columna {}/{}", i + 1, total))?;
    }

    set_progress(&proc, 0.55, "Calculando correlaciones...")?;
    let correlaciones = calcular_correlaciones(&cols_num, umbral);
    set_progress(&proc, 0.75, &format!("{0} pares fuertes", correlaciones.len()))?;

    set_progress(&proc, 0.80, "Agrupando columnas...")?;
    let grupos = agrupar_columnas(&correlaciones, umbral, min_grupo);
    set_progress(&proc, 0.87, &format!("{0} grupos", grupos.len()))?;

    set_progress(&proc, 0.90, "Escribiendo CSVs...")?;
    escribir_perfil_csv(&carpeta_salida.join("perfil_general.csv"), &perfil)?;
    escribir_correlaciones_csv(&carpeta_salida.join("correlaciones_fuertes.csv"), &correlaciones)?;
    escribir_grupos_csv(&carpeta_salida.join("grupos_correlacion.csv"), &grupos)?;
    set_progress(&proc, 1.0, "✅ Finalizado")?;
    Ok(ResultadoPerfil { filas, columnas, carpeta_salida, perfil, correlaciones, grupos })
}

fn set_progress(proc: &Arc<Mutex<ProcState>>, progress: f32, mensaje: &str) -> Result<()> {
    let mut p = proc.lock().unwrap();
    p.progress = progress; p.mensaje = mensaje.to_string();
    Ok(())
}

fn crear_carpeta_salida(path: &Path) -> Result<PathBuf> {
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path.file_stem().and_then(|x| x.to_str()).unwrap_or("resultado");
    let out = base.join(format!("perfil_{name}"));
    fs::create_dir_all(&out)?; Ok(out)
}

fn es_num(dtype: &DataType) -> bool {
    matches!(dtype, DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
        | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
        | DataType::Float32 | DataType::Float64)
}

// ─── Correlaciones ──────────────────────────────────────────────────────────

fn calcular_correlaciones(cols: &[(String, Vec<Option<f64>>)], umbral: f64) -> Vec<Correlacion> {
    let mut r = Vec::new();
    for i in 0..cols.len() {
        for j in (i + 1)..cols.len() {
            if let Some(c) = pearson(&cols[i].1, &cols[j].1) {
                if c.abs() >= umbral {
                    r.push(Correlacion { columna_1: cols[i].0.clone(), columna_2: cols[j].0.clone(),
                        correlacion: c, correlacion_absoluta: c.abs(), fuerza: clasif(c) });
                }
            }
        }
    }
    r.sort_by(|a, b| b.correlacion_absoluta.partial_cmp(&a.correlacion_absoluta).unwrap_or(std::cmp::Ordering::Equal));
    r
}

fn pearson(a: &[Option<f64>], b: &[Option<f64>]) -> Option<f64> {
    let (mut n, mut sx, mut sy, mut sx2, mut sy2, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        if let (Some(x), Some(y)) = (x, y) { if x.is_finite() && y.is_finite() {
            n += 1.0; sx += x; sy += y; sx2 += x*x; sy2 += y*y; sxy += x*y; } }
    }
    if n < 2.0 { return None; }
    let cov = sxy - (sx * sy / n); let vx = sx2 - (sx*sx/n); let vy = sy2 - (sy*sy/n);
    if vx <= 0.0 || vy <= 0.0 { None } else { Some(cov / (vx.sqrt() * vy.sqrt())) }
}

fn clasif(v: f64) -> String {
    let a = v.abs();
    let f = if a >= 0.90 { "muy fuerte" } else if a >= 0.70 { "fuerte" } else if a >= 0.50 { "moderada" } else if a >= 0.30 { "baja" } else { "muy baja" };
    let s = if v > 0.0 { "positiva" } else if v < 0.0 { "negativa" } else { "sin relación" };
    format!("{f} {s}")
}

// ─── Agrupación ─────────────────────────────────────────────────────────────

fn agrupar_columnas(corrs: &[Correlacion], umbral: f64, min_cols: usize) -> Vec<GrupoCorrelacion> {
    let mut adj: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut cmap: HashMap<(&str, &str), f64> = HashMap::new();
    for c in corrs {
        if c.correlacion_absoluta >= umbral {
            adj.entry(c.columna_1.as_str()).or_default().insert(c.columna_2.as_str());
            adj.entry(c.columna_2.as_str()).or_default().insert(c.columna_1.as_str());
            cmap.insert((c.columna_1.as_str(), c.columna_2.as_str()), c.correlacion);
            cmap.insert((c.columna_2.as_str(), c.columna_1.as_str()), c.correlacion);
        }
    }
    let mut vis: HashSet<&str> = HashSet::new();
    let mut grupos = Vec::new();
    for &n in adj.keys() {
        if vis.contains(n) { continue; }
        let mut comp: Vec<&str> = Vec::new();
        let mut q = vec![n]; vis.insert(n);
        while let Some(v) = q.pop() { comp.push(v);
            if let Some(vs) = adj.get(v) { for w in vs { if vis.insert(w) { q.push(w); } } } }
        if comp.len() >= min_cols {
            comp.sort();
            let (mut suma, mut min_r, mut cnt) = (0.0, 1.0, 0u64);
            for i in 0..comp.len() { for j in (i+1)..comp.len() {
                if let Some(&r) = cmap.get(&(comp[i], comp[j])) {
                    let ar = r.abs(); suma += ar; if ar < min_r { min_r = ar; } cnt += 1; } } }
            grupos.push(GrupoCorrelacion {
                columnas: comp.iter().map(|s| s.to_string()).collect(),
                tamano: comp.len(),
                correlacion_minima: min_r,
                correlacion_promedio: if cnt > 0 { suma / cnt as f64 } else { 0.0 },
            });
        }
    }
    grupos.sort_by(|a, b| b.tamano.cmp(&a.tamano));
    grupos
}

// ─── CSV ────────────────────────────────────────────────────────────────────

fn escribir_perfil_csv(path: &Path, p: &[PerfilColumna]) -> Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "columna,tipo,total_filas,no_nulos,nulos,porcentaje_nulos,valores_unicos,media,mediana,minimo,maximo,desviacion_estandar")?;
    for x in p { writeln!(f, "{},{},{},{},{},{:.4},{},{},{},{},{},{}", csv(&x.columna), csv(&x.tipo_polars),
        x.total_filas, x.no_nulos, x.nulos, x.porcentaje_nulos, opt_usize(x.valores_unicos),
        opt_f64(x.media), opt_f64(x.mediana), opt_f64(x.minimo), opt_f64(x.maximo), opt_f64(x.desviacion_estandar))?; }
    Ok(())
}
fn escribir_correlaciones_csv(path: &Path, c: &[Correlacion]) -> Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "columna_1,columna_2,correlacion,correlacion_absoluta,fuerza")?;
    for x in c { writeln!(f, "{},{},{:.8},{:.8},{}", csv(&x.columna_1), csv(&x.columna_2),
        x.correlacion, x.correlacion_absoluta, csv(&x.fuerza))?; }
    Ok(())
}
fn escribir_grupos_csv(path: &Path, g: &[GrupoCorrelacion]) -> Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "grupo,tamano,correlacion_promedio,correlacion_minima,columnas")?;
    for (i, x) in g.iter().enumerate() { writeln!(f, "{},{},{:.6},{:.6},\"{}\"", i+1, x.tamano,
        x.correlacion_promedio, x.correlacion_minima, x.columnas.join("; "))?; }
    Ok(())
}
fn csv(v: &str) -> String { let c = v.replace('"', "\"\""); if c.contains(',')||c.contains('"')||c.contains('\n') { format!("\"{c}\"") } else { c } }
fn opt_f64(v: Option<f64>) -> String { match v { Some(x) if x.is_finite() => format!("{x:.6}"), _ => String::new() } }
fn opt_usize(v: Option<usize>) -> String { v.map(|x| x.to_string()).unwrap_or_default() }

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "perfilador-parquet",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size(egui::vec2(1200.0, 850.0))
                .with_title("📊 Perfilador Parquet"),
            ..Default::default()
        },
        Box::new(|cc| {
            // Tema claro con fondo amarillo/agua
            let mut visuals = egui::Visuals::light();
            visuals.window_fill = A4;
            visuals.panel_fill = A4;
            visuals.faint_bg_color = W4;
            visuals.extreme_bg_color = W4;
            visuals.code_bg_color = W4;
            visuals.widgets.noninteractive.bg_fill = A4;
            visuals.widgets.noninteractive.fg_stroke.color = egui::Color32::from_rgb(40, 80, 90);
            visuals.widgets.active.bg_fill = W2;
            visuals.widgets.hovered.bg_fill = W4;
            visuals.widgets.inactive.bg_fill = A4;
            visuals.selection.bg_fill = W1.gamma_multiply(0.3);
            cc.egui_ctx.set_visuals(visuals);
            Ok(Box::new(PerfilApp {
                archivo: None,
                umbral_texto: "0.70".into(),
                min_grupo_texto: "3".into(),
                proc: Arc::new(Mutex::new(ProcState::default())),
                tab_activa: 0,
            }))
        }),
    )
}
