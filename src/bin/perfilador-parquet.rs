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
enum ProcEstado {
    Idle,
    Processing,
    Done,
    Error(String),
}

#[derive(Clone)]
struct ProcState {
    estado: ProcEstado,
    progress: f32,
    mensaje: String,
    resultado: Option<ResultadoPerfil>,
}

impl Default for ProcState {
    fn default() -> Self {
        Self {
            estado: ProcEstado::Idle,
            progress: 0.0,
            mensaje: "Esperando archivo...".into(),
            resultado: None,
        }
    }
}

#[derive(Default, Clone)]
struct ResultadoPerfil {
    filas: usize,
    columnas: usize,
    carpeta_salida: PathBuf,
    perfil: Vec<PerfilColumna>,
    correlaciones: Vec<Correlacion>,
    grupos: Vec<GrupoCorrelacion>,
    insights: Vec<InsightAuto>,
    recomendaciones: Vec<RecomendacionGrafico>,
    ml: Vec<MetricaMl>,
}

#[derive(Clone)]
struct PerfilColumna {
    columna: String,
    tipo_polars: String,
    tipo_analitico: String,
    total_filas: usize,
    no_nulos: usize,
    nulos: usize,
    porcentaje_nulos: f64,
    valores_unicos: Option<usize>,
    porcentaje_unicos: Option<f64>,
    media: Option<f64>,
    mediana: Option<f64>,
    minimo: Option<f64>,
    maximo: Option<f64>,
    desviacion_estandar: Option<f64>,
    q1: Option<f64>,
    q3: Option<f64>,
    iqr: Option<f64>,
    asimetria: Option<f64>,
    curtosis: Option<f64>,
    ceros_muestra: Option<usize>,
    negativos_muestra: Option<usize>,
    outliers_iqr_muestra: Option<usize>,
    outliers_z_muestra: Option<usize>,
    grafico_sugerido: String,
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

#[derive(Clone)]
struct InsightAuto {
    categoria: String,
    severidad: String,
    columna: String,
    mensaje: String,
}

#[derive(Clone)]
struct RecomendacionGrafico {
    tipo: String,
    columna_x: String,
    columna_y: String,
    prioridad: String,
    razon: String,
}

#[derive(Clone)]
struct MetricaMl {
    columna: String,
    tipo: String,
    score: f64,
    recomendacion: String,
    detalle: String,
}

// ─── Constantes visuales ────────────────────────────────────────────────────

const BORDE_GRUESO: f32 = 2.0;
const ESQUINA: f32 = 8.0;

// Amarillo
const A1: egui::Color32 = egui::Color32::from_rgb(245, 210, 50);
const A2: egui::Color32 = egui::Color32::from_rgb(250, 235, 140);
const A3: egui::Color32 = egui::Color32::from_rgb(200, 160, 30);
const A4: egui::Color32 = egui::Color32::from_rgb(255, 248, 210);

// Agua
const W1: egui::Color32 = egui::Color32::from_rgb(50, 185, 195);
const W2: egui::Color32 = egui::Color32::from_rgb(120, 220, 230);
const W3: egui::Color32 = egui::Color32::from_rgb(25, 140, 155);
const W4: egui::Color32 = egui::Color32::from_rgb(210, 245, 248);

const FONDO_TARJETA: egui::Color32 = A4;
const FONDO_PANEL: egui::Color32 = A4;
const FONDO_GLOBAL: egui::Color32 = W4;
const BORDE: egui::Color32 = egui::Color32::from_rgb(60, 140, 150);

const COLORES_GRUPO: &[egui::Color32] = &[A1, W1, A3, W3, A2, W2];

/// Para archivos grandes evita duplicar millones de valores en memoria.
/// El perfil general usa Polars.
/// Correlaciones, outliers y ML heurístico usan muestra.
const MAX_FILAS_MUESTRA: usize = 200_000;

// ─── Estado de la App ───────────────────────────────────────────────────────

struct PerfilApp {
    archivo: Option<PathBuf>,
    umbral_texto: String,
    min_grupo_texto: String,
    target_texto: String,
    proc: Arc<Mutex<ProcState>>,
    tab_activa: usize,
}

const TAB_LABELS: &[&str] = &[
    "📋 Resumen",
    "📑 Perfil",
    "🔗 Correlaciones",
    "📦 Grupos",
    "🧠 Insights",
    "📈 Gráficos",
    "🤖 ML",
];

impl eframe::App for PerfilApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.painter().rect_filled(ui.max_rect(), 0.0, FONDO_GLOBAL);

        if self.umbral_texto.is_empty() {
            self.umbral_texto = "0.70".into();
        }

        if self.min_grupo_texto.is_empty() {
            self.min_grupo_texto = "3".into();
        }

        let ctx = ui.ctx().clone();
        let proc = self.proc.clone();

        Panel::top("panel_controles").show_inside(ui, |pui| {
            pui.heading("📊 Perfilador de archivos Parquet");
            pui.separator();

            pui.horizontal(|hui| {
                let ocupado = proc.lock().unwrap().estado == ProcEstado::Processing;
                let archivo_ok = self.archivo.is_some();

                if hui
                    .add_enabled(!ocupado, egui::Button::new("📂 Seleccionar archivo"))
                    .on_hover_text("Selecciona un archivo .parquet")
                    .clicked()
                {
                    if let Some(path) = FileDialog::new()
                        .add_filter("Parquet", &["parquet"])
                        .pick_file()
                    {
                        self.archivo = Some(path);

                        let mut p = proc.lock().unwrap();
                        p.estado = ProcEstado::Idle;
                        p.progress = 0.0;
                        p.mensaje = "Archivo seleccionado.".into();
                        p.resultado = None;
                    }
                }

                if hui
                    .add_enabled(archivo_ok && !ocupado, egui::Button::new("⚙️  Procesar"))
                    .on_hover_text("Inicia el perfilamiento y cálculo de correlaciones")
                    .clicked()
                {
                    let path = self.archivo.clone().unwrap();
                    let umbral = self.umbral_texto.trim().parse::<f64>().unwrap_or(0.70);
                    let min_grupo = self.min_grupo_texto.trim().parse::<usize>().unwrap_or(3);
                    let target = self.target_texto.trim().to_string();
                    let pc = proc.clone();

                    {
                        let mut p = proc.lock().unwrap();
                        p.estado = ProcEstado::Processing;
                        p.progress = 0.0;
                        p.mensaje = "Iniciando...".into();
                        p.resultado = None;
                    }

                    thread::spawn(move || {
                        match procesar_parquet(&path, umbral, min_grupo, target, pc.clone()) {
                            Ok(r) => {
                                let mut p = pc.lock().unwrap();
                                p.estado = ProcEstado::Done;
                                p.progress = 1.0;
                                p.mensaje = "✅ Finalizado correctamente.".into();
                                p.resultado = Some(r);
                            }
                            Err(e) => {
                                let mut p = pc.lock().unwrap();
                                p.estado = ProcEstado::Error(format!("{e:#}"));
                                p.mensaje = format!("❌ Error: {e:#}");
                            }
                        }
                    });
                }
            });

            pui.horizontal(|hui| {
                hui.label("🎯 Umbral:");
                hui.add(egui::TextEdit::singleline(&mut self.umbral_texto).desired_width(50.0));

                hui.separator();

                hui.label("📦 Mín. columnas por grupo:");
                hui.add(egui::TextEdit::singleline(&mut self.min_grupo_texto).desired_width(40.0));

                hui.separator();

                hui.label("🤖 Target opcional:");
                hui.add(
                    egui::TextEdit::singleline(&mut self.target_texto)
                        .hint_text("columna objetivo")
                        .desired_width(160.0),
                );
            });

            if let Some(path) = &self.archivo {
                pui.label(format!("📁 {}", path.display()));
            } else {
                pui.label("📁 Ningún archivo seleccionado");
            }

            let p_state = proc.lock().unwrap().clone();

            if p_state.estado == ProcEstado::Processing || p_state.estado == ProcEstado::Idle {
                let bar = egui::ProgressBar::new(p_state.progress)
                    .show_percentage()
                    .animate(p_state.estado == ProcEstado::Processing)
                    .fill(if p_state.estado == ProcEstado::Processing {
                        W1
                    } else {
                        egui::Color32::DARK_GRAY
                    });

                pui.add(bar);
            }

            pui.label(format!("📌 {}", p_state.mensaje));

            if p_state.estado == ProcEstado::Processing {
                ctx.request_repaint_after(Duration::from_millis(50));
            }
        });

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

                            if hui.add(btn).clicked() {
                                self.tab_activa = i;
                            }
                        }
                    });
                });
            });

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
                            4 => tab_insights(sui, &resultado),
                            5 => tab_graficos(sui, &resultado),
                            6 => tab_ml(sui, &resultado),
                            _ => {}
                        }
                    });
                });
            });
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tabs
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
    marco_tab(ui, "📋 Resumen General", |ui| match resultado {
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
                    ui.strong("Métrica");
                    ui.strong("Valor");
                    ui.end_row();

                    ui.label("Filas");
                    ui.label(r.filas.to_string());
                    ui.end_row();

                    ui.label("Columnas");
                    ui.label(r.columnas.to_string());
                    ui.end_row();

                    ui.label("Carpeta de salida");
                    ui.label(r.carpeta_salida.display().to_string());
                    ui.end_row();

                    ui.label("Pares correlacionados");
                    ui.label(r.correlaciones.len().to_string());
                    ui.end_row();

                    ui.label("Grupos encontrados");
                    ui.label(r.grupos.len().to_string());
                    ui.end_row();

                    ui.label("Insights");
                    ui.label(r.insights.len().to_string());
                    ui.end_row();

                    ui.label("Recomendaciones de gráficos");
                    ui.label(r.recomendaciones.len().to_string());
                    ui.end_row();

                    ui.label("Métricas ML");
                    ui.label(r.ml.len().to_string());
                    ui.end_row();
                });

            ui.add_space(16.0);
            ui.separator();

            ui.label("📄 Archivos generados:");

            for f in [
                "perfil_general.csv",
                "correlaciones_fuertes.csv",
                "grupos_correlacion.csv",
                "insights_automaticos.csv",
                "recomendaciones_graficos.csv",
                "ml_sugerencias.csv",
            ] {
                ui.label(format!("   • {f}"));
            }
        }
    });
}

fn tab_perfil(ui: &mut egui::Ui, resultado: &Option<ResultadoPerfil>) {
    marco_tab(ui, "📑 Perfil de Columnas", |ui| match resultado {
        None => {
            ui.add_space(60.0);
            ui.vertical_centered(|ui| {
                ui.heading("Procesa un archivo para ver el perfil.");
            });
        }
        Some(r) => {
            ui.label(format!("Mostrando primeras 200 de {} columnas.", r.perfil.len()));
            ui.add_space(8.0);

            egui::Grid::new("tabla_perfil")
                .striped(true)
                .min_col_width(80.0)
                .show(ui, |ui| {
                    ui.strong("Columna");
                    ui.strong("Tipo");
                    ui.strong("Análisis");
                    ui.strong("Nulos");
                    ui.strong("% Nulos");
                    ui.strong("Únicos");
                    ui.strong("% Únicos");
                    ui.strong("Media");
                    ui.strong("Q1");
                    ui.strong("Mediana");
                    ui.strong("Q3");
                    ui.strong("Mín");
                    ui.strong("Máx");
                    ui.strong("Desv.Est.");
                    ui.strong("Outliers");
                    ui.strong("Gráfico");
                    ui.end_row();

                    for p in r.perfil.iter().take(200) {
                        ui.label(&p.columna);
                        ui.label(&p.tipo_polars);
                        ui.label(&p.tipo_analitico);
                        ui.label(p.nulos.to_string());
                        ui.label(format!("{:.2}", p.porcentaje_nulos));
                        ui.label(opt_usize(p.valores_unicos));
                        ui.label(opt_f64(p.porcentaje_unicos));
                        ui.label(opt_f64(p.media));
                        ui.label(opt_f64(p.q1));
                        ui.label(opt_f64(p.mediana));
                        ui.label(opt_f64(p.q3));
                        ui.label(opt_f64(p.minimo));
                        ui.label(opt_f64(p.maximo));
                        ui.label(opt_f64(p.desviacion_estandar));
                        ui.label(opt_usize(p.outliers_iqr_muestra));
                        ui.label(&p.grafico_sugerido);
                        ui.end_row();
                    }
                });
        }
    });
}

fn tab_correlaciones(ui: &mut egui::Ui, resultado: &Option<ResultadoPerfil>) {
    marco_tab(ui, "🔗 Correlaciones Fuertes", |ui| match resultado {
        None => {
            ui.add_space(60.0);
            ui.vertical_centered(|ui| {
                ui.heading("Procesa un archivo para ver correlaciones.");
            });
        }
        Some(r) => {
            if r.correlaciones.is_empty() {
                ui.label("No se encontraron correlaciones fuertes con el umbral actual.");
                return;
            }

            ui.label(format!(
                "Mostrando primeras 200 de {} pares.",
                r.correlaciones.len()
            ));

            ui.add_space(8.0);

            egui::Grid::new("tabla_correlaciones")
                .striped(true)
                .min_col_width(100.0)
                .show(ui, |ui| {
                    ui.strong("#");
                    ui.strong("Columna 1");
                    ui.strong("Columna 2");
                    ui.strong("Correlación");
                    ui.strong("Fuerza");
                    ui.end_row();

                    for (i, c) in r.correlaciones.iter().take(200).enumerate() {
                        ui.label(format!("{}", i + 1));
                        ui.label(&c.columna_1);
                        ui.label(&c.columna_2);

                        let color = if c.correlacion > 0.0 { W1 } else { A1 };

                        ui.colored_label(color, format!("{:.6}", c.correlacion));
                        ui.label(&c.fuerza);
                        ui.end_row();
                    }
                });
        }
    });
}

fn tab_grupos(ui: &mut egui::Ui, resultado: &Option<ResultadoPerfil>, min_str: &str) {
    marco_tab(ui, "📦 Grupos de Columnas Correlacionadas", |ui| {
        match resultado {
            None => {
                ui.add_space(60.0);
                ui.vertical_centered(|ui| {
                    ui.heading("Procesa un archivo para ver grupos.");
                });
            }
            Some(r) => {
                if r.grupos.is_empty() {
                    ui.label(format!("No se encontraron grupos con ≥ {} columnas.", min_str));
                    return;
                }

                ui.label(format!(
                    "Total: {} grupos (mín. {min_str} columnas)",
                    r.grupos.len()
                ));

                ui.add_space(12.0);

                for (idx, grupo) in r.grupos.iter().enumerate() {
                    let color = COLORES_GRUPO[idx % COLORES_GRUPO.len()];

                    egui::Frame::new()
                        .fill(FONDO_TARJETA)
                        .corner_radius(ESQUINA)
                        .stroke(egui::Stroke::new(BORDE_GRUESO, color))
                        .show(ui, |ui| {
                            ui.horizontal(|hui| {
                                let (rect, _) = hui.allocate_exact_size(
                                    egui::vec2(6.0, 56.0),
                                    egui::Sense::hover(),
                                );

                                hui.painter().rect_filled(rect, 3.0, color);
                                hui.add_space(12.0);

                                hui.vertical(|vui| {
                                    let (rp, rm) = if grupo.tamano > 1 {
                                        (grupo.correlacion_promedio, grupo.correlacion_minima)
                                    } else {
                                        (1.0, 1.0)
                                    };

                                    vui.strong(format!(
                                        "Grupo #{} — {} columnas | r̅ = {:.4} | r_min = {:.4}",
                                        idx + 1,
                                        grupo.tamano,
                                        rp,
                                        rm,
                                    ));

                                    let ancho = 220.0;

                                    let (rect, _) = vui.allocate_exact_size(
                                        egui::vec2(ancho, 16.0),
                                        egui::Sense::hover(),
                                    );

                                    let bg = egui::Rect::from_min_size(
                                        rect.min,
                                        egui::vec2(ancho, 16.0),
                                    );

                                    vui.painter().rect_stroke(
                                        bg,
                                        6.0,
                                        egui::Stroke::new(1.0, BORDE),
                                        egui::StrokeKind::Inside,
                                    );

                                    vui.painter().rect_filled(
                                        bg,
                                        6.0,
                                        egui::Color32::from_gray(35),
                                    );

                                    let fill_w = (rp as f32).clamp(0.0, 1.0) * ancho;

                                    if fill_w > 0.0 {
                                        let fill = egui::Rect::from_min_size(
                                            rect.min,
                                            egui::vec2(fill_w, 16.0),
                                        );

                                        let bar_color = if rp >= 0.9 {
                                            W1
                                        } else if rp >= 0.7 {
                                            A1
                                        } else {
                                            A2
                                        };

                                        vui.painter().rect_filled(fill, 6.0, bar_color);
                                    }

                                    vui.add_space(6.0);

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

fn tab_insights(ui: &mut egui::Ui, resultado: &Option<ResultadoPerfil>) {
    marco_tab(ui, "🧠 Insights Automáticos", |ui| match resultado {
        None => {
            ui.add_space(60.0);
            ui.vertical_centered(|ui| {
                ui.heading("Procesa un archivo para ver insights.");
            });
        }
        Some(r) => {
            if r.insights.is_empty() {
                ui.label(
                    "No se encontraron alertas fuertes. El dataset luce razonable con las reglas actuales.",
                );
                return;
            }

            ui.label(format!("Total: {} insights encontrados.", r.insights.len()));
            ui.add_space(8.0);

            egui::Grid::new("tabla_insights")
                .striped(true)
                .min_col_width(120.0)
                .show(ui, |ui| {
                    ui.strong("Severidad");
                    ui.strong("Categoría");
                    ui.strong("Columna");
                    ui.strong("Mensaje");
                    ui.end_row();

                    for x in r.insights.iter().take(300) {
                        let color = match x.severidad.as_str() {
                            "alta" => A3,
                            "media" => A1,
                            _ => W3,
                        };

                        ui.colored_label(color, &x.severidad);
                        ui.label(&x.categoria);
                        ui.label(&x.columna);
                        ui.label(&x.mensaje);
                        ui.end_row();
                    }
                });
        }
    });
}

fn tab_graficos(ui: &mut egui::Ui, resultado: &Option<ResultadoPerfil>) {
    marco_tab(ui, "📈 Recomendaciones de Gráficos", |ui| match resultado {
        None => {
            ui.add_space(60.0);
            ui.vertical_centered(|ui| {
                ui.heading("Procesa un archivo para ver gráficos recomendados.");
            });
        }
        Some(r) => {
            if r.recomendaciones.is_empty() {
                ui.label("No se generaron recomendaciones de gráficos.");
                return;
            }

            ui.label(format!(
                "Mostrando primeras 300 de {} recomendaciones.",
                r.recomendaciones.len()
            ));

            ui.add_space(8.0);

            egui::Grid::new("tabla_graficos")
                .striped(true)
                .min_col_width(120.0)
                .show(ui, |ui| {
                    ui.strong("Prioridad");
                    ui.strong("Gráfico");
                    ui.strong("Columna X");
                    ui.strong("Columna Y");
                    ui.strong("Razón");
                    ui.end_row();

                    for x in r.recomendaciones.iter().take(300) {
                        ui.label(&x.prioridad);
                        ui.label(&x.tipo);
                        ui.label(&x.columna_x);
                        ui.label(&x.columna_y);
                        ui.label(&x.razon);
                        ui.end_row();
                    }
                });
        }
    });
}

fn tab_ml(ui: &mut egui::Ui, resultado: &Option<ResultadoPerfil>) {
    marco_tab(ui, "🤖 Sugerencias tipo Machine Learning", |ui| match resultado {
        None => {
            ui.add_space(60.0);
            ui.vertical_centered(|ui| {
                ui.heading("Procesa un archivo para ver sugerencias ML.");
            });
        }
        Some(r) => {
            if r.ml.is_empty() {
                ui.label("No se generaron métricas ML.");
                return;
            }

            ui.label("El ML aquí es liviano y explicable: puntúa columnas, detecta columnas candidatas y, si se escribe un target numérico, calcula relevancia por correlación con ese objetivo.");

            ui.add_space(8.0);

            egui::Grid::new("tabla_ml")
                .striped(true)
                .min_col_width(120.0)
                .show(ui, |ui| {
                    ui.strong("Score");
                    ui.strong("Columna");
                    ui.strong("Tipo");
                    ui.strong("Recomendación");
                    ui.strong("Detalle");
                    ui.end_row();

                    for x in r.ml.iter().take(300) {
                        let color = if x.score >= 0.75 {
                            W3
                        } else if x.score >= 0.50 {
                            A3
                        } else {
                            egui::Color32::DARK_RED
                        };

                        ui.colored_label(color, format!("{:.3}", x.score));
                        ui.label(&x.columna);
                        ui.label(&x.tipo);
                        ui.label(&x.recomendacion);
                        ui.label(&x.detalle);
                        ui.end_row();
                    }
                });
        }
    });
}

// ═════════════════════════════════════════════════════════════════════════════
// Lógica de procesamiento
// ═════════════════════════════════════════════════════════════════════════════

fn procesar_parquet(
    path: &Path,
    umbral: f64,
    min_grupo: usize,
    target_col: String,
    proc: Arc<Mutex<ProcState>>,
) -> Result<ResultadoPerfil> {
    set_progress(&proc, 0.02, "Abriendo archivo Parquet...")?;

    let mut file = File::open(path)
        .with_context(|| format!("No se pudo abrir {}", path.display()))?;

    set_progress(&proc, 0.05, "Leyendo Parquet...")?;

    let df = ParquetReader::new(&mut file)
        .finish()
        .context("No se pudo leer Parquet")?;

    let filas = df.height();
    let columnas = df.width();

    set_progress(
        &proc,
        0.10,
        &format!("Leído: {filas} filas x {columnas} columnas"),
    )?;

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

        let pct_n = if t > 0 {
            (nulos as f64 / t as f64) * 100.0
        } else {
            0.0
        };

        let unicos = s.n_unique().ok();

        let pct_unicos = unicos.map(|u| {
            if t > 0 {
                (u as f64 / t as f64) * 100.0
            } else {
                0.0
            }
        });

        let tipo_analitico = tipo_analitico_columna(s, unicos, t);
        let grafico_sugerido = grafico_sugerido_columna(&tipo_analitico, unicos, t);

        let mut media = None;
        let mut mediana = None;
        let mut minimo = None;
        let mut maximo = None;
        let mut desv = None;
        let mut q1 = None;
        let mut q3 = None;
        let mut iqr = None;
        let mut asimetria = None;
        let mut curtosis = None;
        let mut ceros = None;
        let mut negativos = None;
        let mut out_iqr = None;
        let mut out_z = None;

        if es_num(s.dtype()) {
            if let Ok(f64s) = s.cast(&DataType::Float64) {
                media = f64s.mean();
                mediana = f64s.median();
                desv = f64s.std(1);

                minimo = f64s.min::<f64>().ok().flatten();
                maximo = f64s.max::<f64>().ok().flatten();

                if let Ok(ca) = f64s.f64() {
                    let muestra_opts = muestrear_f64(ca, MAX_FILAS_MUESTRA);

                    let muestra_vals: Vec<f64> = muestra_opts
                        .iter()
                        .filter_map(|x| *x)
                        .filter(|x| x.is_finite())
                        .collect();

                    cols_num.push((nombre.clone(), muestra_opts));

                    let stats = stats_muestra(&muestra_vals);

                    q1 = stats.q1;
                    q3 = stats.q3;
                    iqr = stats.iqr;
                    asimetria = stats.asimetria;
                    curtosis = stats.curtosis;

                    ceros = Some(muestra_vals.iter().filter(|x| **x == 0.0).count());
                    negativos = Some(muestra_vals.iter().filter(|x| **x < 0.0).count());

                    out_iqr = stats.outliers_iqr;
                    out_z = stats.outliers_z;
                }
            }
        }

        perfil.push(PerfilColumna {
            columna: nombre,
            tipo_polars: tipo,
            tipo_analitico,
            total_filas: t,
            no_nulos,
            nulos,
            porcentaje_nulos: pct_n,
            valores_unicos: unicos,
            porcentaje_unicos: pct_unicos,
            media,
            mediana,
            minimo,
            maximo,
            desviacion_estandar: desv,
            q1,
            q3,
            iqr,
            asimetria,
            curtosis,
            ceros_muestra: ceros,
            negativos_muestra: negativos,
            outliers_iqr_muestra: out_iqr,
            outliers_z_muestra: out_z,
            grafico_sugerido,
        });

        let pct = 0.10 + (i as f64 / total as f64) * 0.40;

        set_progress(
            &proc,
            pct as f32,
            &format!("Perfilando columna {}/{}", i + 1, total),
        )?;
    }

    set_progress(&proc, 0.55, "Calculando correlaciones...")?;

    let correlaciones = calcular_correlaciones(&cols_num, umbral);

    set_progress(
        &proc,
        0.75,
        &format!("{0} pares fuertes", correlaciones.len()),
    )?;

    set_progress(&proc, 0.80, "Agrupando columnas...")?;

    let grupos = agrupar_columnas(&correlaciones, umbral, min_grupo);

    set_progress(&proc, 0.87, &format!("{0} grupos", grupos.len()))?;

    set_progress(&proc, 0.89, "Generando insights y recomendaciones...")?;

    let insights = generar_insights(&perfil, &correlaciones, &grupos);
    let recomendaciones = recomendar_graficos(&perfil, &correlaciones);
    let ml = calcular_ml_columnas(&perfil, &cols_num, &target_col);

    set_progress(&proc, 0.92, "Escribiendo CSVs...")?;

    escribir_perfil_csv(&carpeta_salida.join("perfil_general.csv"), &perfil)?;
    escribir_correlaciones_csv(
        &carpeta_salida.join("correlaciones_fuertes.csv"),
        &correlaciones,
    )?;
    escribir_grupos_csv(&carpeta_salida.join("grupos_correlacion.csv"), &grupos)?;
    escribir_insights_csv(&carpeta_salida.join("insights_automaticos.csv"), &insights)?;
    escribir_recomendaciones_csv(
        &carpeta_salida.join("recomendaciones_graficos.csv"),
        &recomendaciones,
    )?;
    escribir_ml_csv(&carpeta_salida.join("ml_sugerencias.csv"), &ml)?;

    set_progress(&proc, 1.0, "✅ Finalizado")?;

    Ok(ResultadoPerfil {
        filas,
        columnas,
        carpeta_salida,
        perfil,
        correlaciones,
        grupos,
        insights,
        recomendaciones,
        ml,
    })
}

fn set_progress(proc: &Arc<Mutex<ProcState>>, progress: f32, mensaje: &str) -> Result<()> {
    let mut p = proc.lock().unwrap();
    p.progress = progress;
    p.mensaje = mensaje.to_string();
    Ok(())
}

fn crear_carpeta_salida(path: &Path) -> Result<PathBuf> {
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or("resultado");

    let out = base.join(format!("perfil_{name}"));

    fs::create_dir_all(&out)?;

    Ok(out)
}

fn es_num(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

fn tipo_analitico_columna(s: &Series, unicos: Option<usize>, total: usize) -> String {
    let dtype_txt = format!("{:?}", s.dtype()).to_lowercase();

    if es_num(s.dtype()) {
        return "numérica".into();
    }

    if dtype_txt.contains("bool") {
        return "booleana".into();
    }

    if dtype_txt.contains("date") || dtype_txt.contains("time") {
        return "temporal".into();
    }

    let ratio = unicos
        .map(|u| if total > 0 { u as f64 / total as f64 } else { 0.0 })
        .unwrap_or(1.0);

    if ratio <= 0.20 || unicos.unwrap_or(usize::MAX) <= 50 {
        "categórica".into()
    } else {
        "texto/id".into()
    }
}

fn grafico_sugerido_columna(tipo: &str, unicos: Option<usize>, total: usize) -> String {
    match tipo {
        "numérica" => "histograma + boxplot".into(),
        "temporal" => "línea temporal".into(),
        "booleana" => "barras".into(),
        "categórica" => {
            let u = unicos.unwrap_or(0);

            if u <= 20 {
                "barras".into()
            } else {
                "barras top 20".into()
            }
        }
        _ => {
            let u = unicos.unwrap_or(total);

            if u == total {
                "no graficar: posible ID".into()
            } else {
                "frecuencia top 20".into()
            }
        }
    }
}

fn muestrear_f64(ca: &Float64Chunked, max: usize) -> Vec<Option<f64>> {
    let len = ca.len();

    if len <= max {
        return ca.into_iter().collect();
    }

    let step = ((len as f64 / max as f64).ceil() as usize).max(1);

    ca.into_iter()
        .enumerate()
        .filter_map(|(i, v)| if i % step == 0 { Some(v) } else { None })
        .take(max)
        .collect()
}

#[derive(Default)]
struct StatsMuestra {
    q1: Option<f64>,
    q3: Option<f64>,
    iqr: Option<f64>,
    asimetria: Option<f64>,
    curtosis: Option<f64>,
    outliers_iqr: Option<usize>,
    outliers_z: Option<usize>,
}

fn stats_muestra(v: &[f64]) -> StatsMuestra {
    if v.len() < 4 {
        return StatsMuestra::default();
    }

    let mut sorted = v.to_vec();

    sorted.sort_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let q1 = percentil_sorted(&sorted, 0.25);
    let q3 = percentil_sorted(&sorted, 0.75);
    let iqr = q3 - q1;

    let low = q1 - 1.5 * iqr;
    let high = q3 + 1.5 * iqr;

    let out_iqr = if iqr > 0.0 {
        Some(v.iter().filter(|x| **x < low || **x > high).count())
    } else {
        Some(0)
    };

    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
    let std = var.sqrt();

    let (asim, curt, out_z) = if std > 0.0 {
        let z: Vec<f64> = v.iter().map(|x| (x - mean) / std).collect();

        let asim = z.iter().map(|x| x.powi(3)).sum::<f64>() / z.len() as f64;
        let curt = z.iter().map(|x| x.powi(4)).sum::<f64>() / z.len() as f64 - 3.0;
        let out_z = z.iter().filter(|x| x.abs() >= 3.0).count();

        (Some(asim), Some(curt), Some(out_z))
    } else {
        (None, None, Some(0))
    };

    StatsMuestra {
        q1: Some(q1),
        q3: Some(q3),
        iqr: Some(iqr),
        asimetria: asim,
        curtosis: curt,
        outliers_iqr: out_iqr,
        outliers_z: out_z,
    }
}

fn percentil_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }

    let pos = p.clamp(0.0, 1.0) * (sorted.len().saturating_sub(1)) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;

    if lo == hi {
        sorted[lo]
    } else {
        let w = pos - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

// ─── Correlaciones ──────────────────────────────────────────────────────────

fn calcular_correlaciones(cols: &[(String, Vec<Option<f64>>)], umbral: f64) -> Vec<Correlacion> {
    let mut r = Vec::new();

    for i in 0..cols.len() {
        for j in (i + 1)..cols.len() {
            if let Some(c) = pearson(&cols[i].1, &cols[j].1) {
                if c.abs() >= umbral {
                    r.push(Correlacion {
                        columna_1: cols[i].0.clone(),
                        columna_2: cols[j].0.clone(),
                        correlacion: c,
                        correlacion_absoluta: c.abs(),
                        fuerza: clasif(c),
                    });
                }
            }
        }
    }

    r.sort_by(|a, b| {
        b.correlacion_absoluta
            .partial_cmp(&a.correlacion_absoluta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    r
}

fn pearson(a: &[Option<f64>], b: &[Option<f64>]) -> Option<f64> {
    let (mut n, mut sx, mut sy, mut sx2, mut sy2, mut sxy) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    for (x, y) in a.iter().zip(b) {
        if let (Some(x), Some(y)) = (x, y) {
            if x.is_finite() && y.is_finite() {
                n += 1.0;
                sx += x;
                sy += y;
                sx2 += x * x;
                sy2 += y * y;
                sxy += x * y;
            }
        }
    }

    if n < 2.0 {
        return None;
    }

    let cov = sxy - (sx * sy / n);
    let vx = sx2 - (sx * sx / n);
    let vy = sy2 - (sy * sy / n);

    if vx <= 0.0 || vy <= 0.0 {
        None
    } else {
        Some(cov / (vx.sqrt() * vy.sqrt()))
    }
}

fn clasif(v: f64) -> String {
    let a = v.abs();

    let f = if a >= 0.90 {
        "muy fuerte"
    } else if a >= 0.70 {
        "fuerte"
    } else if a >= 0.50 {
        "moderada"
    } else if a >= 0.30 {
        "baja"
    } else {
        "muy baja"
    };

    let s = if v > 0.0 {
        "positiva"
    } else if v < 0.0 {
        "negativa"
    } else {
        "sin relación"
    };

    format!("{f} {s}")
}

// ─── Agrupación ─────────────────────────────────────────────────────────────

fn agrupar_columnas(
    corrs: &[Correlacion],
    umbral: f64,
    min_cols: usize,
) -> Vec<GrupoCorrelacion> {
    let mut adj: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut cmap: HashMap<(&str, &str), f64> = HashMap::new();

    for c in corrs {
        if c.correlacion_absoluta >= umbral {
            adj.entry(c.columna_1.as_str())
                .or_default()
                .insert(c.columna_2.as_str());

            adj.entry(c.columna_2.as_str())
                .or_default()
                .insert(c.columna_1.as_str());

            cmap.insert((c.columna_1.as_str(), c.columna_2.as_str()), c.correlacion);
            cmap.insert((c.columna_2.as_str(), c.columna_1.as_str()), c.correlacion);
        }
    }

    let mut vis: HashSet<&str> = HashSet::new();
    let mut grupos = Vec::new();

    for &n in adj.keys() {
        if vis.contains(n) {
            continue;
        }

        let mut comp: Vec<&str> = Vec::new();
        let mut q = vec![n];

        vis.insert(n);

        while let Some(v) = q.pop() {
            comp.push(v);

            if let Some(vs) = adj.get(v) {
                for w in vs {
                    if vis.insert(w) {
                        q.push(w);
                    }
                }
            }
        }

        if comp.len() >= min_cols {
            comp.sort();

            let (mut suma, mut min_r, mut cnt) = (0.0, 1.0, 0u64);

            for i in 0..comp.len() {
                for j in (i + 1)..comp.len() {
                    if let Some(&r) = cmap.get(&(comp[i], comp[j])) {
                        let ar = r.abs();

                        suma += ar;

                        if ar < min_r {
                            min_r = ar;
                        }

                        cnt += 1;
                    }
                }
            }

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

// ─── Insights, gráficos y ML liviano ────────────────────────────────────────

fn generar_insights(
    perfil: &[PerfilColumna],
    corrs: &[Correlacion],
    grupos: &[GrupoCorrelacion],
) -> Vec<InsightAuto> {
    let mut out = Vec::new();

    for p in perfil {
        if p.porcentaje_nulos >= 50.0 {
            out.push(InsightAuto {
                categoria: "calidad".into(),
                severidad: "alta".into(),
                columna: p.columna.clone(),
                mensaje: format!(
                    "Tiene {:.2}% de nulos; puede afectar modelos, filtros y gráficos.",
                    p.porcentaje_nulos
                ),
            });
        } else if p.porcentaje_nulos >= 20.0 {
            out.push(InsightAuto {
                categoria: "calidad".into(),
                severidad: "media".into(),
                columna: p.columna.clone(),
                mensaje: format!(
                    "Tiene {:.2}% de nulos; conviene revisar imputación o exclusión.",
                    p.porcentaje_nulos
                ),
            });
        }

        if p.valores_unicos == Some(1) {
            out.push(InsightAuto {
                categoria: "variabilidad".into(),
                severidad: "media".into(),
                columna: p.columna.clone(),
                mensaje: "Columna constante; no aporta a correlaciones ni modelos.".into(),
            });
        }

        if let Some(pu) = p.porcentaje_unicos {
            if pu >= 95.0 && p.total_filas > 100 {
                out.push(InsightAuto {
                    categoria: "identificador".into(),
                    severidad: "baja".into(),
                    columna: p.columna.clone(),
                    mensaje: format!(
                        "{:.2}% de valores únicos; posiblemente sea ID y no conviene graficarla como categoría.",
                        pu
                    ),
                });
            }
        }

        if let Some(outliers) = p.outliers_iqr_muestra {
            let denom = p.no_nulos.min(MAX_FILAS_MUESTRA).max(1);
            let pct = (outliers as f64 / denom as f64) * 100.0;

            if pct >= 5.0 {
                out.push(InsightAuto {
                    categoria: "outliers".into(),
                    severidad: "media".into(),
                    columna: p.columna.clone(),
                    mensaje: format!(
                        "La muestra tiene {:.2}% de outliers por IQR; revisar con boxplot.",
                        pct
                    ),
                });
            }
        }

        if let Some(asim) = p.asimetria {
            if asim.abs() >= 2.0 {
                out.push(InsightAuto {
                    categoria: "distribución".into(),
                    severidad: "baja".into(),
                    columna: p.columna.clone(),
                    mensaje: format!(
                        "Distribución muy asimétrica (skew {:.2}); conviene histograma o escala log si aplica.",
                        asim
                    ),
                });
            }
        }
    }

    for c in corrs.iter().take(200) {
        if c.correlacion_absoluta >= 0.95 {
            out.push(InsightAuto {
                categoria: "redundancia".into(),
                severidad: "media".into(),
                columna: format!("{} ↔ {}", c.columna_1, c.columna_2),
                mensaje: format!(
                    "Correlación casi duplicada ({:.4}); en ML podría eliminarse una de las dos.",
                    c.correlacion
                ),
            });
        }
    }

    for (i, g) in grupos.iter().take(50).enumerate() {
        out.push(InsightAuto {
            categoria: "grupo".into(),
            severidad: "baja".into(),
            columna: format!("Grupo {}", i + 1),
            mensaje: format!(
                "{} columnas se mueven juntas; revisar si representan el mismo fenómeno.",
                g.tamano
            ),
        });
    }

    out
}

fn recomendar_graficos(
    perfil: &[PerfilColumna],
    corrs: &[Correlacion],
) -> Vec<RecomendacionGrafico> {
    let mut out = Vec::new();

    for p in perfil {
        let prioridad =
            if p.tipo_analitico == "numérica" && p.outliers_iqr_muestra.unwrap_or(0) > 0 {
                "alta"
            } else {
                "media"
            };

        out.push(RecomendacionGrafico {
            tipo: p.grafico_sugerido.clone(),
            columna_x: p.columna.clone(),
            columna_y: String::new(),
            prioridad: prioridad.into(),
            razon: format!(
                "Tipo detectado: {}; nulos: {:.2}%; únicos: {}",
                p.tipo_analitico,
                p.porcentaje_nulos,
                opt_usize(p.valores_unicos)
            ),
        });
    }

    for c in corrs.iter().take(200) {
        out.push(RecomendacionGrafico {
            tipo: "scatter plot".into(),
            columna_x: c.columna_1.clone(),
            columna_y: c.columna_2.clone(),
            prioridad: if c.correlacion_absoluta >= 0.90 {
                "alta".into()
            } else {
                "media".into()
            },
            razon: format!(
                "Relación {} con r={:.4}; sirve para validar tendencia y posibles duplicidades.",
                c.fuerza,
                c.correlacion
            ),
        });
    }

    out.sort_by(|a, b| prioridad_val(&b.prioridad).cmp(&prioridad_val(&a.prioridad)));

    out
}

fn prioridad_val(x: &str) -> i32 {
    match x {
        "alta" => 3,
        "media" => 2,
        _ => 1,
    }
}

fn calcular_ml_columnas(
    perfil: &[PerfilColumna],
    cols_num: &[(String, Vec<Option<f64>>)],
    target_col: &str,
) -> Vec<MetricaMl> {
    let target = target_col.trim();
    let mut out = Vec::new();

    if !target.is_empty() {
        if let Some((_, y)) = cols_num
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(target))
        {
            for (nombre, x) in cols_num {
                if nombre.eq_ignore_ascii_case(target) {
                    continue;
                }

                if let Some(r) = pearson(x, y) {
                    out.push(MetricaMl {
                        columna: nombre.clone(),
                        tipo: "relevancia contra target".into(),
                        score: r.abs().clamp(0.0, 1.0),
                        recomendacion: if r.abs() >= 0.70 {
                            "variable fuerte para modelo".into()
                        } else if r.abs() >= 0.30 {
                            "variable candidata".into()
                        } else {
                            "baja relación lineal".into()
                        },
                        detalle: format!("Correlación con target '{}': {:.4}", target, r),
                    });
                }
            }

            out.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            return out;
        } else {
            out.push(MetricaMl {
                columna: target.into(),
                tipo: "target".into(),
                score: 0.0,
                recomendacion: "target no encontrado o no numérico".into(),
                detalle: "Para relevancia supervisada, el target debe existir y ser numérico en esta versión liviana.".into(),
            });
        }
    }

    for p in perfil {
        let completitud = 1.0 - (p.porcentaje_nulos / 100.0).clamp(0.0, 1.0);

        let variabilidad = match p.valores_unicos {
            Some(0) | Some(1) => 0.0,
            Some(u) if u <= 50 && p.tipo_analitico == "categórica" => 0.75,
            Some(_) if p.tipo_analitico == "numérica" => 1.0,
            Some(_) => 0.45,
            None => 0.35,
        };

        let id_penalty = if p.porcentaje_unicos.unwrap_or(0.0) >= 95.0 {
            0.35
        } else {
            1.0
        };

        let score = (0.65 * completitud + 0.35 * variabilidad) * id_penalty;

        let rec = if score >= 0.75 {
            "buena variable candidata"
        } else if score >= 0.50 {
            "usar con limpieza previa"
        } else {
            "baja utilidad para ML"
        };

        out.push(MetricaMl {
            columna: p.columna.clone(),
            tipo: p.tipo_analitico.clone(),
            score: score.clamp(0.0, 1.0),
            recomendacion: rec.into(),
            detalle: format!(
                "completitud={:.2}, únicos={}, gráfico={}",
                completitud,
                opt_usize(p.valores_unicos),
                p.grafico_sugerido
            ),
        });
    }

    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    out
}

// ─── CSV ────────────────────────────────────────────────────────────────────

fn escribir_perfil_csv(path: &Path, p: &[PerfilColumna]) -> Result<()> {
    let mut f = File::create(path)?;

    writeln!(
        f,
        "columna,tipo_polars,tipo_analitico,total_filas,no_nulos,nulos,porcentaje_nulos,valores_unicos,porcentaje_unicos,media,q1,mediana,q3,minimo,maximo,desviacion_estandar,iqr,asimetria,curtosis,ceros_muestra,negativos_muestra,outliers_iqr_muestra,outliers_z_muestra,grafico_sugerido"
    )?;

    for x in p {
        writeln!(
            f,
            "{},{},{},{},{},{},{:.4},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            csv(&x.columna),
            csv(&x.tipo_polars),
            csv(&x.tipo_analitico),
            x.total_filas,
            x.no_nulos,
            x.nulos,
            x.porcentaje_nulos,
            opt_usize(x.valores_unicos),
            opt_f64(x.porcentaje_unicos),
            opt_f64(x.media),
            opt_f64(x.q1),
            opt_f64(x.mediana),
            opt_f64(x.q3),
            opt_f64(x.minimo),
            opt_f64(x.maximo),
            opt_f64(x.desviacion_estandar),
            opt_f64(x.iqr),
            opt_f64(x.asimetria),
            opt_f64(x.curtosis),
            opt_usize(x.ceros_muestra),
            opt_usize(x.negativos_muestra),
            opt_usize(x.outliers_iqr_muestra),
            opt_usize(x.outliers_z_muestra),
            csv(&x.grafico_sugerido)
        )?;
    }

    Ok(())
}

fn escribir_correlaciones_csv(path: &Path, c: &[Correlacion]) -> Result<()> {
    let mut f = File::create(path)?;

    writeln!(
        f,
        "columna_1,columna_2,correlacion,correlacion_absoluta,fuerza"
    )?;

    for x in c {
        writeln!(
            f,
            "{},{},{:.8},{:.8},{}",
            csv(&x.columna_1),
            csv(&x.columna_2),
            x.correlacion,
            x.correlacion_absoluta,
            csv(&x.fuerza)
        )?;
    }

    Ok(())
}

fn escribir_grupos_csv(path: &Path, g: &[GrupoCorrelacion]) -> Result<()> {
    let mut f = File::create(path)?;

    writeln!(
        f,
        "grupo,tamano,correlacion_promedio,correlacion_minima,columnas"
    )?;

    for (i, x) in g.iter().enumerate() {
        writeln!(
            f,
            "{},{},{:.6},{:.6},\"{}\"",
            i + 1,
            x.tamano,
            x.correlacion_promedio,
            x.correlacion_minima,
            x.columnas.join("; ")
        )?;
    }

    Ok(())
}

fn escribir_insights_csv(path: &Path, xs: &[InsightAuto]) -> Result<()> {
    let mut f = File::create(path)?;

    writeln!(f, "severidad,categoria,columna,mensaje")?;

    for x in xs {
        writeln!(
            f,
            "{},{},{},{}",
            csv(&x.severidad),
            csv(&x.categoria),
            csv(&x.columna),
            csv(&x.mensaje)
        )?;
    }

    Ok(())
}

fn escribir_recomendaciones_csv(path: &Path, xs: &[RecomendacionGrafico]) -> Result<()> {
    let mut f = File::create(path)?;

    writeln!(f, "prioridad,tipo_grafico,columna_x,columna_y,razon")?;

    for x in xs {
        writeln!(
            f,
            "{},{},{},{},{}",
            csv(&x.prioridad),
            csv(&x.tipo),
            csv(&x.columna_x),
            csv(&x.columna_y),
            csv(&x.razon)
        )?;
    }

    Ok(())
}

fn escribir_ml_csv(path: &Path, xs: &[MetricaMl]) -> Result<()> {
    let mut f = File::create(path)?;

    writeln!(f, "score,columna,tipo,recomendacion,detalle")?;

    for x in xs {
        writeln!(
            f,
            "{:.6},{},{},{},{}",
            x.score,
            csv(&x.columna),
            csv(&x.tipo),
            csv(&x.recomendacion),
            csv(&x.detalle)
        )?;
    }

    Ok(())
}

fn csv(v: &str) -> String {
    let c = v.replace('"', "\"\"");

    if c.contains(',') || c.contains('"') || c.contains('\n') {
        format!("\"{c}\"")
    } else {
        c
    }
}

fn opt_f64(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.6}"),
        _ => String::new(),
    }
}

fn opt_usize(v: Option<usize>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}

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
            let mut visuals = egui::Visuals::light();

            visuals.window_fill = A4;
            visuals.panel_fill = A4;
            visuals.faint_bg_color = W4;
            visuals.extreme_bg_color = W4;
            visuals.code_bg_color = W4;
            visuals.widgets.noninteractive.bg_fill = A4;
            visuals.widgets.noninteractive.fg_stroke.color =
                egui::Color32::from_rgb(40, 80, 90);
            visuals.widgets.active.bg_fill = W2;
            visuals.widgets.hovered.bg_fill = W4;
            visuals.widgets.inactive.bg_fill = A4;
            visuals.selection.bg_fill = W1.gamma_multiply(0.3);

            cc.egui_ctx.set_visuals(visuals);

            Ok(Box::new(PerfilApp {
                archivo: None,
                umbral_texto: "0.70".into(),
                min_grupo_texto: "3".into(),
                target_texto: String::new(),
                proc: Arc::new(Mutex::new(ProcState::default())),
                tab_activa: 0,
            }))
        }),
    )
}
