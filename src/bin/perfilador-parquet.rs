// ============================================================================
// perfilador-parquet — Perfilador de archivos Parquet con egui + Polars
// Lee un .parquet, genera perfil de columnas, calcula correlaciones Pearson,
// agrupa columnas correlacionadas (mín. 3), muestra en GUI y exporta CSV.
// Incluye barra de progreso real con hilos.
// Ejecutar: cargo run --bin perfilador-parquet
// ============================================================================

use anyhow::{Context, Result};
use eframe::egui;
use polars::prelude::*;
use rfd::FileDialog;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
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
    kmeans: Option<KMeansResultado>,
    montecarlo: Option<MonteCarloResultado>,
    feature_selection: Option<FeatureSelectionResultado>,
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

#[derive(Clone)]
struct KMeansResultado {
    features: Vec<String>,
    k: usize,
    filas_muestra: usize,
    clusters: Vec<KMeansCluster>,
    outliers: Vec<KMeansOutlier>,
    distancia_promedio: f64,
    distancia_p95: f64,
    distancia_p99: f64,
    conclusion: String,
}

#[derive(Clone)]
struct KMeansCluster {
    cluster: usize,
    cantidad: usize,
    porcentaje: f64,
    distancia_promedio: f64,
    lectura: String,
}

#[derive(Clone)]
struct KMeansOutlier {
    ranking: usize,
    fila_muestra: usize,
    fila_aproximada: usize,
    cluster: usize,
    distancia: f64,
    z_score: f64,
    severidad: String,
    razon: String,
    valores_principales: String,
}

#[derive(Clone)]
struct MonteCarloResultado {
    simulaciones: usize,
    tamano_lote: usize,
    columnas: Vec<MonteCarloColumna>,
    escenarios: Vec<MonteCarloEscenario>,
    riesgos: Vec<MonteCarloRiesgo>,
    conclusion: String,
}

#[derive(Clone)]
struct MonteCarloColumna {
    columna: String,
    muestras_validas: usize,
    promedio_historico: f64,
    desviacion_historica: f64,
    minimo_historico: f64,
    p5_historico: f64,
    p50_historico: f64,
    p95_historico: f64,
    maximo_historico: f64,
    promedio_sim_p5: f64,
    promedio_sim_p50: f64,
    promedio_sim_p95: f64,
    total_sim_p5: f64,
    total_sim_p50: f64,
    total_sim_p95: f64,
    prob_superar_p95: f64,
    prob_bajo_p5: f64,
    lectura: String,
}

#[derive(Clone)]
struct MonteCarloEscenario {
    columna: String,
    escenario: String,
    promedio_estimado: f64,
    total_lote_estimado: f64,
    interpretacion: String,
}

#[derive(Clone)]
struct MonteCarloRiesgo {
    columna: String,
    severidad: String,
    indicador: String,
    valor: f64,
    lectura: String,
}

#[derive(Clone)]
struct FeatureSelectionResultado {
    columnas_originales: usize,
    columnas_recomendadas: usize,
    columnas_descartadas: usize,
    columnas_revision: usize,
    aplicado_a_analisis: bool,
    umbral_score: f64,
    columnas_usadas_analisis: Vec<String>,
    columnas: Vec<FeatureSelectionColumna>,
    conclusion: String,
}

#[derive(Clone)]
struct FeatureSelectionColumna {
    columna: String,
    decision: String,
    score: f64,
    tipo_analitico: String,
    porcentaje_nulos: f64,
    porcentaje_unicos: Option<f64>,
    correlacion_maxima: Option<f64>,
    correlacion_con: String,
    motivo: String,
    accion: String,
    riesgo: String,
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
/// KMeans usa una muestra menor para mantener la interfaz rápida y estable.
const MAX_FILAS_KMEANS: usize = 25_000;
const MAX_COLUMNAS_KMEANS: usize = 12;
const MAX_COLUMNAS_MONTECARLO: usize = 8;
const MONTECARLO_SIMULACIONES: usize = 5_000;
const MONTECARLO_TAMANO_LOTE: usize = 250;

const APP_VERSION: &str = "Entregable 1";
const CLAVE_ACCESO: &str = "cr19780302";

// ─── Estado de la App ───────────────────────────────────────────────────────

struct PerfilApp {
    acceso_autorizado: bool,
    clave_ingresada: String,
    mensaje_acceso: String,
    archivo: Option<PathBuf>,
    carpeta_salida_base: Option<PathBuf>,
    umbral_texto: String,
    min_grupo_texto: String,
    target_texto: String,
    aplicar_feature_selection: bool,
    min_score_feature_texto: String,
    proc: Arc<Mutex<ProcState>>,
    tab_activa: usize,
}

const TAB_LABELS: &[&str] = &[
    "📋 Resumen",
    "🧭 Hallazgos",
    "🧹 Feature Selection",
    "📑 Perfil",
    "🔗 Correlaciones",
    "📦 Grupos",
    "🧠 Insights",
    "📈 Gráficos",
    "🧩 KMeans",
    "🎲 Monte Carlo",
    "🤖 ML",
];

impl eframe::App for PerfilApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.umbral_texto.is_empty() {
            self.umbral_texto = "0.70".into();
        }

        if self.min_grupo_texto.is_empty() {
            self.min_grupo_texto = "3".into();
        }

        if self.min_score_feature_texto.is_empty() {
            self.min_score_feature_texto = "0.55".into();
        }

        let ctx = ui.ctx().clone();

        if !self.acceso_autorizado {
            self.ui_login(&ctx);
            return;
        }

        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(FONDO_GLOBAL)
                    .inner_margin(14.0),
            )
            .show_inside(ui, |ui| {
                self.ui_header(ui);
                ui.add_space(10.0);
                self.ui_controles(ui, &ctx);
                ui.add_space(12.0);
                self.ui_tabs(ui);
            });
    }
}

impl PerfilApp {
    fn ui_login(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(FONDO_GLOBAL)
                    .inner_margin(24.0),
            )
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(120.0);
                    egui::Frame::new()
                        .fill(FONDO_TARJETA)
                        .stroke(egui::Stroke::new(2.0, BORDE))
                        .corner_radius(14.0)
                        .inner_margin(24.0)
                        .show(ui, |ui| {
                            ui.set_width(430.0);
                            ui.vertical_centered(|ui| {
                                ui.heading(format!("🔐 Perfilador Parquet — {APP_VERSION}"));
                                ui.add_space(6.0);
                                ui.label("Ingrese la clave de uso para habilitar el análisis.");
                                ui.add_space(16.0);

                                let respuesta = ui.add(
                                    egui::TextEdit::singleline(&mut self.clave_ingresada)
                                        .password(true)
                                        .hint_text("Clave de acceso")
                                        .desired_width(260.0),
                                );

                                let enter = respuesta.lost_focus()
                                    && ui.input(|i| i.key_pressed(egui::Key::Enter));

                                ui.add_space(8.0);
                                let ingresar = ui
                                    .add(egui::Button::new("Ingresar").min_size(egui::vec2(140.0, 34.0)))
                                    .clicked();

                                if ingresar || enter {
                                    if self.clave_ingresada.trim() == CLAVE_ACCESO {
                                        self.acceso_autorizado = true;
                                        self.mensaje_acceso.clear();
                                        self.clave_ingresada.clear();
                                    } else {
                                        self.mensaje_acceso = "Clave incorrecta. Verifique e intente nuevamente.".into();
                                    }
                                }

                                if !self.mensaje_acceso.is_empty() {
                                    ui.add_space(8.0);
                                    ui.colored_label(egui::Color32::DARK_RED, &self.mensaje_acceso);
                                }
                            });
                        });
                });
            });
    }

    fn ui_header(&self, ui: &mut egui::Ui) {
        egui::Frame::new()
            .fill(FONDO_TARJETA)
            .stroke(egui::Stroke::new(1.5, BORDE))
            .corner_radius(12.0)
            .inner_margin(14.0)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.vertical(|ui| {
                        ui.heading(format!("📊 Perfilador Parquet Inteligente — {APP_VERSION}"));
                        ui.label("Carga un archivo Parquet, selecciona la salida y genera perfil, Feature Selection, correlaciones, hallazgos, KMeans, Monte Carlo, ML liviano y reporte HTML.");
                    });
                });
            });
    }

    fn ui_controles(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let proc = self.proc.clone();
        let ocupado = { proc.lock().unwrap().estado == ProcEstado::Processing };
        let archivo_ok = self.archivo.is_some();

        egui::Frame::new()
            .fill(FONDO_PANEL)
            .stroke(egui::Stroke::new(1.5, BORDE))
            .corner_radius(12.0)
            .inner_margin(12.0)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(!ocupado, egui::Button::new("📂 Seleccionar Parquet").min_size(egui::vec2(170.0, 34.0)))
                        .on_hover_text("Selecciona el archivo .parquet que se va a analizar")
                        .clicked()
                    {
                        if let Some(path) = FileDialog::new()
                            .add_filter("Parquet", &["parquet"])
                            .pick_file()
                        {
                            if self.carpeta_salida_base.is_none() {
                                self.carpeta_salida_base = path.parent().map(|p| p.to_path_buf());
                            }

                            self.archivo = Some(path);

                            let mut p = proc.lock().unwrap();
                            p.estado = ProcEstado::Idle;
                            p.progress = 0.0;
                            p.mensaje = "Archivo seleccionado. Puede procesarse cuando la carpeta de salida esté correcta.".into();
                            p.resultado = None;
                        }
                    }

                    if ui
                        .add_enabled(!ocupado, egui::Button::new("📁 Elegir salida").min_size(egui::vec2(145.0, 34.0)))
                        .on_hover_text("Selecciona dónde se guardará la carpeta perfil_<archivo>")
                        .clicked()
                    {
                        if let Some(path) = FileDialog::new().pick_folder() {
                            self.carpeta_salida_base = Some(path);

                            let mut p = proc.lock().unwrap();
                            if p.estado != ProcEstado::Processing {
                                p.mensaje = "Carpeta de salida seleccionada.".into();
                            }
                        }
                    }

                    if ui
                        .add_enabled(archivo_ok && !ocupado, egui::Button::new("⚙️ Procesar y generar archivos").min_size(egui::vec2(225.0, 34.0)))
                        .on_hover_text("Genera CSV, reporte HTML, Feature Selection, insights, KMeans, Monte Carlo y recomendaciones visuales")
                        .clicked()
                    {
                        let path = self.archivo.clone().unwrap();
                        let umbral = self.umbral_texto.trim().parse::<f64>().unwrap_or(0.70);
                        let min_grupo = self.min_grupo_texto.trim().parse::<usize>().unwrap_or(3);
                        let target = self.target_texto.trim().to_string();
                        let aplicar_feature_selection = self.aplicar_feature_selection;
                        let min_score_feature = self.min_score_feature_texto.trim().parse::<f64>().unwrap_or(0.55).clamp(0.0, 1.0);
                        let carpeta_salida_base = self.carpeta_salida_base.clone();
                        let pc = proc.clone();

                        {
                            let mut p = proc.lock().unwrap();
                            p.estado = ProcEstado::Processing;
                            p.progress = 0.0;
                            p.mensaje = "Iniciando análisis...".into();
                            p.resultado = None;
                        }

                        thread::spawn(move || {
                            match procesar_parquet(&path, umbral, min_grupo, target, aplicar_feature_selection, min_score_feature, carpeta_salida_base, pc.clone()) {
                                Ok(r) => {
                                    let mut p = pc.lock().unwrap();
                                    p.estado = ProcEstado::Done;
                                    p.progress = 1.0;
                                    p.mensaje = format!("✅ Finalizado. Archivos guardados en {}", r.carpeta_salida.display());
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

                ui.add_space(10.0);
                ui.columns(2, |cols| {
                    ui_path_card(&mut cols[0], "Archivo Parquet", self.archivo.as_ref(), "Ningún archivo seleccionado");
                    ui_path_card(&mut cols[1], "Salida de reportes", self.carpeta_salida_base.as_ref(), "Se usará la carpeta del Parquet");
                });

                ui.add_space(10.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label("🎯 Umbral correlación:");
                    ui.add(egui::TextEdit::singleline(&mut self.umbral_texto).desired_width(55.0));

                    ui.separator();

                    ui.label("📦 Mínimo por grupo:");
                    ui.add(egui::TextEdit::singleline(&mut self.min_grupo_texto).desired_width(45.0));

                    ui.separator();

                    ui.label("🤖 Target opcional:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.target_texto)
                            .hint_text("columna objetivo")
                            .desired_width(180.0),
                    );

                    ui.separator();

                    ui.checkbox(
                        &mut self.aplicar_feature_selection,
                        "🧹 Aplicar Feature Selection antes de análisis pesados",
                    )
                    .on_hover_text("Reduce columnas para correlaciones, KMeans, Monte Carlo y ML. El perfil general siempre se calcula completo.");

                    ui.label("Score mínimo:");
                    ui.add(egui::TextEdit::singleline(&mut self.min_score_feature_texto).desired_width(48.0));
                });

                ui.add_space(10.0);
                let p_state = proc.lock().unwrap().clone();
                let color_estado = match p_state.estado {
                    ProcEstado::Done => W3,
                    ProcEstado::Error(_) => egui::Color32::DARK_RED,
                    ProcEstado::Processing => W1,
                    ProcEstado::Idle => egui::Color32::DARK_GRAY,
                };

                ui.horizontal(|ui| {
                    ui.colored_label(color_estado, "●");
                    ui.label(&p_state.mensaje);
                });

                ui.add(
                    egui::ProgressBar::new(p_state.progress)
                        .show_percentage()
                        .animate(p_state.estado == ProcEstado::Processing)
                        .fill(if p_state.estado == ProcEstado::Processing { W1 } else { color_estado }),
                );

                if let Some(r) = &p_state.resultado {
                    ui.add_space(8.0);
                    ui.columns(7, |cols| {
                        ui_kpi(&mut cols[0], "Filas", &r.filas.to_string());
                        ui_kpi(&mut cols[1], "Columnas", &r.columnas.to_string());
                        let fs_ok = r.feature_selection.as_ref().map(|f| f.columnas_recomendadas).unwrap_or(0);
                        ui_kpi(&mut cols[2], "Útiles", &fs_ok.to_string());
                        ui_kpi(&mut cols[3], "Correlaciones", &r.correlaciones.len().to_string());
                        ui_kpi(&mut cols[4], "Insights", &r.insights.len().to_string());
                        let outliers = r.kmeans.as_ref().map(|k| k.outliers.len()).unwrap_or(0);
                        ui_kpi(&mut cols[5], "Raros KMeans", &outliers.to_string());
                        let mc = r.montecarlo.as_ref().map(|m| m.columnas.len()).unwrap_or(0);
                        ui_kpi(&mut cols[6], "Monte Carlo", &mc.to_string());
                    });
                }

                if p_state.estado == ProcEstado::Processing {
                    ctx.request_repaint_after(Duration::from_millis(50));
                }
            });
    }

    fn ui_tabs(&mut self, ui: &mut egui::Ui) {
        egui::Frame::new()
            .fill(FONDO_GLOBAL)
            .stroke(egui::Stroke::new(BORDE_GRUESO, BORDE))
            .corner_radius(12.0)
            .inner_margin(8.0)
            .show(ui, |ui| {
                egui::ScrollArea::horizontal()
                    .id_salt("tabs_scroll_horizontal")
                    .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for (i, label) in TAB_LABELS.iter().enumerate() {
                            let selected = self.tab_activa == i;
                            let mut btn = egui::Button::selectable(selected, *label).min_size(egui::vec2(118.0, 32.0));

                            if selected {
                                btn = btn.fill(W1);
                                btn = btn.stroke(egui::Stroke::new(2.0, A1));
                            } else {
                                btn = btn.stroke(egui::Stroke::new(1.0, BORDE));
                            }

                            if ui.add(btn).clicked() {
                                self.tab_activa = i;
                            }
                        }
                    });
                });
            });

        ui.add_space(8.0);

        egui::Frame::new()
            .fill(FONDO_GLOBAL)
            .stroke(egui::Stroke::new(BORDE_GRUESO, BORDE))
            .corner_radius(12.0)
            .inner_margin(8.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("contenido_tab_scroll_vertical")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                    let resultado = self.proc.lock().unwrap().resultado.clone();

                    match self.tab_activa {
                        0 => tab_resumen(ui, &resultado),
                        1 => tab_hallazgos(ui, &resultado),
                        2 => tab_feature_selection(ui, &resultado),
                        3 => tab_perfil(ui, &resultado),
                        4 => tab_correlaciones(ui, &resultado),
                        5 => tab_grupos(ui, &resultado, &self.min_grupo_texto),
                        6 => tab_insights(ui, &resultado),
                        7 => tab_graficos(ui, &resultado),
                        8 => tab_kmeans(ui, &resultado),
                        9 => tab_montecarlo(ui, &resultado),
                        10 => tab_ml(ui, &resultado),
                        _ => {}
                    }
                });
            });
    }
}

fn ui_path_card(ui: &mut egui::Ui, titulo: &str, path: Option<&PathBuf>, vacio: &str) {
    egui::Frame::new()
        .fill(W4)
        .stroke(egui::Stroke::new(1.0, BORDE))
        .corner_radius(10.0)
        .inner_margin(10.0)
        .show(ui, |ui| {
            ui.strong(titulo);
            ui.add_space(4.0);
            match path {
                Some(p) => {
                    ui.monospace(p.display().to_string());
                }
                None => {
                    ui.colored_label(egui::Color32::DARK_GRAY, vacio);
                }
            }
        });
}

fn ui_kpi(ui: &mut egui::Ui, titulo: &str, valor: &str) {
    egui::Frame::new()
        .fill(W4)
        .stroke(egui::Stroke::new(1.0, BORDE))
        .corner_radius(10.0)
        .inner_margin(10.0)
        .show(ui, |ui| {
            ui.label(titulo);
            ui.heading(valor);
        });
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

                    ui.label("Feature Selection");
                    let fs_txt = r.feature_selection.as_ref()
                        .map(|f| format!("{} útiles / {} revisar / {} descartadas", f.columnas_recomendadas, f.columnas_revision, f.columnas_descartadas))
                        .unwrap_or_else(|| "no aplicado".into());
                    ui.label(fs_txt);
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

                    ui.label("KMeans / registros raros");
                    let km = r.kmeans.as_ref()
                        .map(|k| format!("{} clusters / {} raros", k.k, k.outliers.len()))
                        .unwrap_or_else(|| "no aplicado".into());
                    ui.label(km);
                    ui.end_row();

                    ui.label("Monte Carlo");
                    let mc = r.montecarlo.as_ref()
                        .map(|m| format!("{} simulaciones / {} columnas", m.simulaciones, m.columnas.len()))
                        .unwrap_or_else(|| "no aplicado".into());
                    ui.label(mc);
                    ui.end_row();
                });

            ui.add_space(12.0);
            ui.horizontal(|ui| {
                if ui.button("📂 Abrir carpeta de salida").clicked() {
                    abrir_carpeta_en_sistema(&r.carpeta_salida);
                }
                ui.monospace(r.carpeta_salida.display().to_string());
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
                "feature_selection.csv",
                "kmeans_resumen.csv",
                "kmeans_outliers.csv",
                "montecarlo_resumen.csv",
                "montecarlo_escenarios.csv",
                "montecarlo_riesgos.csv",
                "reporte_analitico.html",
            ] {
                ui.label(format!("   • {f}"));
            }
        }
    });
}


fn tab_hallazgos(ui: &mut egui::Ui, resultado: &Option<ResultadoPerfil>) {
    marco_tab(ui, "🧭 Hallazgos para revisar primero", |ui| match resultado {
        None => {
            ui.add_space(60.0);
            ui.vertical_centered(|ui| {
                ui.heading("Procesa un archivo para ver hallazgos explicados en lenguaje simple.");
            });
        }
        Some(r) => {
            ui.label("Esta pantalla está pensada para una persona no experta: muestra qué debería revisarse primero y por qué podría importar.");
            ui.add_space(12.0);

            let alta = r.insights.iter().filter(|x| x.severidad == "alta").count();
            let media = r.insights.iter().filter(|x| x.severidad == "media").count();
            let raros = r.kmeans.as_ref().map(|k| k.outliers.len()).unwrap_or(0);
            let mc_cols = r.montecarlo.as_ref().map(|m| m.columnas.len()).unwrap_or(0);

            ui.columns(5, |cols| {
                ui_kpi(&mut cols[0], "Alertas altas", &alta.to_string());
                ui_kpi(&mut cols[1], "Alertas medias", &media.to_string());
                ui_kpi(&mut cols[2], "Correlaciones", &r.correlaciones.len().to_string());
                ui_kpi(&mut cols[3], "Registros raros", &raros.to_string());
                ui_kpi(&mut cols[4], "Simuladas", &mc_cols.to_string());
            });

            ui.add_space(16.0);
            ui.heading("🔎 Lectura rápida");

            if let Some(fs) = &r.feature_selection {
                card_hallazgo(
                    ui,
                    "🧹 Columnas realmente útiles",
                    if fs.columnas_descartadas > fs.columnas_recomendadas { "media" } else { "baja" },
                    &fs.conclusion,
                    "Abrir la pestaña Feature Selection y revisar feature_selection.csv antes de hacer dashboards o modelos.",
                );
            }

            if let Some(km) = &r.kmeans {
                card_hallazgo(
                    ui,
                    "🧩 Registros que no se parecen al resto",
                    if km.outliers.is_empty() { "baja" } else { "alta" },
                    &km.conclusion,
                    "Abrir la pestaña KMeans y revisar las primeras filas del archivo kmeans_outliers.csv.",
                );
            } else {
                card_hallazgo(
                    ui,
                    "🧩 KMeans no aplicado",
                    "baja",
                    "No hubo suficientes columnas numéricas limpias para formar clusters confiables.",
                    "No es un error. En archivos principalmente textuales conviene usar perfil, nulos, únicos y frecuencia de categorías.",
                );
            }

            if let Some(mc) = &r.montecarlo {
                card_hallazgo(
                    ui,
                    "🎲 Escenarios futuros por Monte Carlo",
                    if mc.riesgos.iter().any(|x| x.severidad == "alta") { "alta" } else { "media" },
                    &mc.conclusion,
                    "Abrir la pestaña Monte Carlo y revisar montecarlo_resumen.csv para ver escenarios optimista, probable y pesimista.",
                );
            }

            if let Some(c) = r.correlaciones.first() {
                card_hallazgo(
                    ui,
                    "🔗 Relación fuerte entre columnas",
                    if c.correlacion_absoluta >= 0.95 { "alta" } else { "media" },
                    &format!("{} y {} se mueven casi juntas con r={:.4}.", c.columna_1, c.columna_2, c.correlacion),
                    "Puede significar duplicidad, dependencia fuerte o una regla de negocio escondida. Conviene graficarlas como scatter plot.",
                );
            }

            if let Some(g) = r.grupos.first() {
                card_hallazgo(
                    ui,
                    "📦 Bloque de columnas relacionadas",
                    "media",
                    &format!("Se detectó un grupo de {} columnas que se comportan parecido.", g.tamano),
                    "Cuando varias columnas se mueven juntas, puede existir información repetida o un fenómeno común que no se ve a simple vista.",
                );
            }

            for x in r.insights.iter().take(12) {
                card_hallazgo(
                    ui,
                    &format!("{} — {}", emoji_categoria(&x.categoria), x.categoria),
                    &x.severidad,
                    &format!("{}: {}", x.columna, x.mensaje),
                    "Revisar la columna indicada antes de construir dashboards o modelos.",
                );
            }
        }
    });
}

fn card_hallazgo(ui: &mut egui::Ui, titulo: &str, severidad: &str, mensaje: &str, accion: &str) {
    let color = match severidad {
        "alta" => egui::Color32::from_rgb(180, 80, 40),
        "media" => A3,
        _ => W3,
    };

    egui::Frame::new()
        .fill(FONDO_TARJETA)
        .stroke(egui::Stroke::new(1.8, color))
        .corner_radius(10.0)
        .inner_margin(12.0)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(color, "●");
                ui.strong(titulo);
                ui.label(format!("({})", severidad));
            });
            ui.add_space(4.0);
            ui.label(mensaje);
            ui.add_space(4.0);
            ui.colored_label(egui::Color32::DARK_GRAY, format!("Qué hacer: {accion}"));
        });
    ui.add_space(8.0);
}

fn emoji_categoria(categoria: &str) -> &'static str {
    match categoria {
        "calidad" => "🧹",
        "outliers" => "⚠️",
        "redundancia" => "🔁",
        "grupo" => "📦",
        "kmeans" => "🧩",
        "segmento pequeño" => "🧬",
        "montecarlo" => "🎲",
        "riesgo" => "🚦",
        "feature_selection" => "🧹",
        "identificador" => "🪪",
        "distribución" => "📉",
        _ => "🔎",
    }
}


fn tab_feature_selection(ui: &mut egui::Ui, resultado: &Option<ResultadoPerfil>) {
    marco_tab(ui, "🧹 Feature Selection: selección inteligente de columnas", |ui| match resultado {
        None => {
            ui.add_space(60.0);
            ui.vertical_centered(|ui| {
                ui.heading("Procesa un archivo para ver qué columnas conviene usar o descartar.");
            });
        }
        Some(r) => match &r.feature_selection {
            None => {
                ui.label("No se generó Feature Selection.");
            }
            Some(fs) => {
                ui.label("Esta pestaña sirve para reducir ruido antes de correlaciones, KMeans, Monte Carlo y ML. No borra columnas del archivo original: solo recomienda cuáles usar para análisis.");
                ui.add_space(10.0);

                ui.columns(5, |cols| {
                    ui_kpi(&mut cols[0], "Originales", &fs.columnas_originales.to_string());
                    ui_kpi(&mut cols[1], "Usar", &fs.columnas_recomendadas.to_string());
                    ui_kpi(&mut cols[2], "Revisar", &fs.columnas_revision.to_string());
                    ui_kpi(&mut cols[3], "Descartar", &fs.columnas_descartadas.to_string());
                    ui_kpi(&mut cols[4], "Usadas análisis", &fs.columnas_usadas_analisis.len().to_string());
                });

                ui.add_space(10.0);
                card_hallazgo(
                    ui,
                    "Lectura simple",
                    if fs.columnas_descartadas > fs.columnas_recomendadas { "media" } else { "baja" },
                    &fs.conclusion,
                    "Las columnas marcadas como Usar son mejores candidatas. Las de Revisar pueden servir, pero tienen algún riesgo. Las de Descartar suelen ser IDs, constantes, muy nulas o redundantes.",
                );

                ui.add_space(10.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!("Aplicado a análisis pesados: {}", if fs.aplicado_a_analisis { "sí" } else { "no" }));
                    ui.separator();
                    ui.label(format!("Score mínimo: {:.2}", fs.umbral_score));
                });

                ui.add_space(10.0);
                egui::Grid::new("tabla_feature_selection")
                    .striped(true)
                    .min_col_width(90.0)
                    .show(ui, |ui| {
                        ui.strong("Decisión");
                        ui.strong("Score");
                        ui.strong("Columna");
                        ui.strong("Tipo");
                        ui.strong("% Nulos");
                        ui.strong("% Únicos");
                        ui.strong("Corr. máx.");
                        ui.strong("Con");
                        ui.strong("Riesgo");
                        ui.strong("Motivo");
                        ui.strong("Acción");
                        ui.end_row();

                        for x in fs.columnas.iter().take(400) {
                            let color = match x.decision.as_str() {
                                "Usar" => W3,
                                "Revisar" => A3,
                                _ => egui::Color32::DARK_RED,
                            };

                            ui.colored_label(color, &x.decision);
                            ui.label(format!("{:.3}", x.score));
                            ui.label(&x.columna);
                            ui.label(&x.tipo_analitico);
                            ui.label(format!("{:.2}", x.porcentaje_nulos));
                            ui.label(opt_f64(x.porcentaje_unicos));
                            ui.label(opt_f64(x.correlacion_maxima));
                            ui.label(&x.correlacion_con);
                            ui.label(&x.riesgo);
                            ui.label(&x.motivo);
                            ui.label(&x.accion);
                            ui.end_row();
                        }
                    });
            }
        },
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
    marco_tab(ui, "📈 Gráficos e Inteligencia Visual", |ui| match resultado {
        None => {
            ui.add_space(60.0);
            ui.vertical_centered(|ui| {
                ui.heading("Procesa un archivo para ver gráficos recomendados.");
            });
        }
        Some(r) => {
            ui.label("Vista rápida para descubrir problemas, columnas candidatas, correlaciones y gráficos útiles antes de construir tableros.");
            ui.add_space(10.0);

            let top_nulos: Vec<(String, f64)> = r.perfil.iter()
                .filter(|p| p.porcentaje_nulos > 0.0)
                .map(|p| (p.columna.clone(), p.porcentaje_nulos))
                .collect();

            let top_unicos: Vec<(String, f64)> = r.perfil.iter()
                .filter_map(|p| p.porcentaje_unicos.map(|v| (p.columna.clone(), v)))
                .collect();

            ui.columns(2, |cols| {
                draw_horizontal_bar_chart(&mut cols[0], "Top columnas con más nulos (%)", &top_nulos, 8, 100.0);
                draw_horizontal_bar_chart(&mut cols[1], "Top columnas con más valores únicos (%)", &top_unicos, 8, 100.0);
            });

            ui.add_space(14.0);
            draw_correlation_heatmap(ui, "Mapa visual de correlaciones fuertes", &r.correlaciones, 12);

            ui.add_space(14.0);
            draw_ml_score_chart(ui, "Ranking visual ML / utilidad analítica", &r.ml, 10);

            ui.add_space(14.0);
            ui.separator();
            ui.heading("📌 Recomendaciones automáticas de gráficos");

            if r.recomendaciones.is_empty() {
                ui.label("No se generaron recomendaciones de gráficos.");
                return;
            }

            ui.label(format!("Mostrando primeras 300 de {} recomendaciones.", r.recomendaciones.len()));
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
                        let color = match x.prioridad.as_str() {
                            "alta" => W3,
                            "media" => A3,
                            _ => egui::Color32::DARK_GRAY,
                        };
                        ui.colored_label(color, &x.prioridad);
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

fn draw_horizontal_bar_chart(ui: &mut egui::Ui, titulo: &str, datos: &[(String, f64)], limite: usize, maximo: f64) {
    egui::Frame::new()
        .fill(FONDO_TARJETA)
        .stroke(egui::Stroke::new(1.0, BORDE))
        .corner_radius(8.0)
        .show(ui, |ui| {
            ui.strong(titulo);
            ui.add_space(8.0);

            let mut ordenados = datos.to_vec();
            ordenados.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            if ordenados.is_empty() {
                ui.label("Sin datos para graficar.");
                return;
            }

            let ancho = ui.available_width().max(260.0);
            let alto_fila = 26.0;
            let total_h = (ordenados.iter().take(limite).count() as f32 * alto_fila).max(40.0);
            let (rect, _) = ui.allocate_exact_size(egui::vec2(ancho, total_h), egui::Sense::hover());
            let painter = ui.painter_at(rect);

            let label_w = (ancho * 0.42).clamp(120.0, 260.0);
            let bar_x = rect.left() + label_w;
            let bar_w = (ancho - label_w - 52.0).max(80.0);

            for (idx, (nombre, valor)) in ordenados.iter().take(limite).enumerate() {
                let y = rect.top() + idx as f32 * alto_fila;
                let pct = (*valor / maximo).clamp(0.0, 1.0) as f32;
                let fondo = egui::Rect::from_min_size(egui::pos2(bar_x, y + 5.0), egui::vec2(bar_w, 14.0));
                let barra = egui::Rect::from_min_size(egui::pos2(bar_x, y + 5.0), egui::vec2(bar_w * pct, 14.0));

                painter.text(egui::pos2(rect.left(), y + 2.0), egui::Align2::LEFT_TOP, abreviar(nombre, 28), egui::FontId::monospace(11.0), egui::Color32::DARK_GRAY);
                painter.rect_filled(fondo, 4.0, egui::Color32::from_gray(220));
                painter.rect_filled(barra, 4.0, if *valor >= 50.0 { A3 } else { W3 });
                painter.text(egui::pos2(bar_x + bar_w + 6.0, y + 2.0), egui::Align2::LEFT_TOP, format!("{:.1}", valor), egui::FontId::monospace(11.0), egui::Color32::DARK_GRAY);
            }
        });
}

fn draw_correlation_heatmap(ui: &mut egui::Ui, titulo: &str, corrs: &[Correlacion], limite: usize) {
    egui::Frame::new()
        .fill(FONDO_TARJETA)
        .stroke(egui::Stroke::new(1.0, BORDE))
        .corner_radius(8.0)
        .show(ui, |ui| {
            ui.strong(titulo);
            ui.add_space(8.0);

            if corrs.is_empty() {
                ui.label("No hay correlaciones fuertes con el umbral actual.");
                return;
            }

            let mut columnas: Vec<String> = Vec::new();
            for c in corrs.iter().take(300) {
                if !columnas.contains(&c.columna_1) { columnas.push(c.columna_1.clone()); }
                if !columnas.contains(&c.columna_2) { columnas.push(c.columna_2.clone()); }
                if columnas.len() >= limite { break; }
            }

            let n = columnas.len().min(limite);
            if n < 2 {
                ui.label("Se necesitan al menos dos columnas correlacionadas.");
                return;
            }

            let celda = 26.0;
            let label_w = 165.0;
            let ancho = label_w + n as f32 * celda + 8.0;
            let alto = 34.0 + n as f32 * celda;
            let (rect, _) = ui.allocate_exact_size(egui::vec2(ancho, alto), egui::Sense::hover());
            let painter = ui.painter_at(rect);
            let start_x = rect.left() + label_w;
            let start_y = rect.top() + 30.0;

            for (i, col) in columnas.iter().take(n).enumerate() {
                painter.text(egui::pos2(start_x + i as f32 * celda + 2.0, rect.top() + 4.0), egui::Align2::LEFT_TOP, format!("{}", i + 1), egui::FontId::monospace(10.0), egui::Color32::DARK_GRAY);
                painter.text(egui::pos2(rect.left(), start_y + i as f32 * celda + 5.0), egui::Align2::LEFT_TOP, format!("{} {}", i + 1, abreviar(col, 22)), egui::FontId::monospace(10.5), egui::Color32::DARK_GRAY);
            }

            for i in 0..n {
                for j in 0..n {
                    let mut valor = if i == j { 1.0 } else { 0.0 };
                    if i != j {
                        for c in corrs {
                            let ok = (c.columna_1 == columnas[i] && c.columna_2 == columnas[j]) || (c.columna_1 == columnas[j] && c.columna_2 == columnas[i]);
                            if ok { valor = c.correlacion; break; }
                        }
                    }

                    let intensidad = valor.abs().clamp(0.0, 1.0);
                    let color = if valor >= 0.0 {
                        egui::Color32::from_rgb((220.0 - intensidad * 70.0) as u8, 245, 248)
                    } else {
                        egui::Color32::from_rgb(255, (248.0 - intensidad * 80.0) as u8, 210)
                    };

                    let cell = egui::Rect::from_min_size(egui::pos2(start_x + j as f32 * celda, start_y + i as f32 * celda), egui::vec2(celda - 2.0, celda - 2.0));
                    painter.rect_filled(cell, 3.0, color);
                    painter.rect_stroke(cell, 3.0, egui::Stroke::new(0.5, BORDE), egui::StrokeKind::Inside);
                }
            }

            ui.add_space(4.0);
            ui.label("Celeste = correlación positiva. Amarillo = correlación negativa. Más intenso = relación más fuerte.");
        });
}

fn draw_ml_score_chart(ui: &mut egui::Ui, titulo: &str, ml: &[MetricaMl], limite: usize) {
    let datos: Vec<(String, f64)> = ml.iter().map(|m| (format!("{} ({})", m.columna, m.tipo), m.score * 100.0)).collect();
    draw_horizontal_bar_chart(ui, titulo, &datos, limite, 100.0);
}

fn abreviar(s: &str, max: usize) -> String {
    if s.chars().count() <= max { return s.to_string(); }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}


fn tab_kmeans(ui: &mut egui::Ui, resultado: &Option<ResultadoPerfil>) {
    marco_tab(ui, "🧩 KMeans: grupos naturales y registros raros", |ui| match resultado {
        None => {
            ui.add_space(60.0);
            ui.vertical_centered(|ui| {
                ui.heading("Procesa un archivo para ver clusters y registros raros.");
            });
        }
        Some(r) => {
            let Some(km) = &r.kmeans else {
                ui.label("KMeans no se aplicó porque no hubo suficientes columnas numéricas útiles. Esto puede pasar cuando el archivo tiene muchas columnas de texto, IDs o columnas con demasiados nulos.");
                return;
            };

            ui.label("KMeans agrupa filas parecidas. Las filas muy alejadas de su grupo aparecen como posibles anomalías. No significa automáticamente error: significa que merecen revisión.");
            ui.add_space(10.0);

            ui.columns(4, |cols| {
                ui_kpi(&mut cols[0], "Clusters", &km.k.to_string());
                ui_kpi(&mut cols[1], "Filas muestra", &km.filas_muestra.to_string());
                ui_kpi(&mut cols[2], "Variables usadas", &km.features.len().to_string());
                ui_kpi(&mut cols[3], "Raros", &km.outliers.len().to_string());
            });

            ui.add_space(10.0);
            egui::Frame::new()
                .fill(W4)
                .stroke(egui::Stroke::new(1.0, BORDE))
                .corner_radius(8.0)
                .inner_margin(10.0)
                .show(ui, |ui| {
                    ui.strong("Lectura automática");
                    ui.label(&km.conclusion);
                    ui.add_space(4.0);
                    ui.label(format!("Variables usadas: {}", km.features.join(", ")));
                    ui.label(format!("Distancia promedio: {:.4} | p95: {:.4} | p99: {:.4}", km.distancia_promedio, km.distancia_p95, km.distancia_p99));
                });

            ui.add_space(14.0);
            let datos_cluster: Vec<(String, f64)> = km.clusters.iter()
                .map(|c| (format!("Cluster {}", c.cluster), c.porcentaje))
                .collect();
            draw_horizontal_bar_chart(ui, "Tamaño de clusters (%)", &datos_cluster, 12, 100.0);

            ui.add_space(14.0);
            ui.heading("📦 Resumen de clusters");
            egui::Grid::new("tabla_kmeans_clusters")
                .striped(true)
                .min_col_width(120.0)
                .show(ui, |ui| {
                    ui.strong("Cluster");
                    ui.strong("Cantidad");
                    ui.strong("%");
                    ui.strong("Distancia prom.");
                    ui.strong("Lectura");
                    ui.end_row();

                    for c in &km.clusters {
                        ui.label(c.cluster.to_string());
                        ui.label(c.cantidad.to_string());
                        ui.label(format!("{:.2}", c.porcentaje));
                        ui.label(format!("{:.4}", c.distancia_promedio));
                        ui.label(&c.lectura);
                        ui.end_row();
                    }
                });

            ui.add_space(14.0);
            ui.heading("⚠️ Registros más raros de la muestra");

            if km.outliers.is_empty() {
                ui.label("No se detectaron registros suficientemente alejados de su cluster.");
                return;
            }

            ui.label("La fila aproximada se calcula sobre la muestra. Para auditoría exacta conviene cruzar con un ID o exportar la fila real en una siguiente versión.");
            ui.add_space(8.0);

            egui::Grid::new("tabla_kmeans_outliers")
                .striped(true)
                .min_col_width(100.0)
                .show(ui, |ui| {
                    ui.strong("#");
                    ui.strong("Fila aprox.");
                    ui.strong("Cluster");
                    ui.strong("Distancia");
                    ui.strong("Z");
                    ui.strong("Severidad");
                    ui.strong("Valores que explican rareza");
                    ui.end_row();

                    for x in km.outliers.iter().take(100) {
                        let color = if x.severidad == "alta" { egui::Color32::DARK_RED } else { A3 };
                        ui.label(x.ranking.to_string());
                        ui.label(x.fila_aproximada.to_string());
                        ui.label(x.cluster.to_string());
                        ui.label(format!("{:.4}", x.distancia));
                        ui.label(format!("{:.2}", x.z_score));
                        ui.colored_label(color, &x.severidad);
                        ui.label(&x.valores_principales);
                        ui.end_row();
                    }
                });
        }
    });
}

fn tab_montecarlo(ui: &mut egui::Ui, resultado: &Option<ResultadoPerfil>) {
    marco_tab(ui, "🎲 Monte Carlo: escenarios probables y riesgos", |ui| match resultado {
        None => {
            ui.add_space(60.0);
            ui.vertical_centered(|ui| {
                ui.heading("Procesa un archivo para simular escenarios con Monte Carlo.");
            });
        }
        Some(r) => {
            let Some(mc) = &r.montecarlo else {
                ui.label("Monte Carlo no se aplicó porque no hubo columnas numéricas útiles. Conviene usarlo en montos, tiempos, costos, edades, cantidades, consumos o duraciones.");
                return;
            };

            ui.label("Monte Carlo genera miles de escenarios posibles usando el comportamiento histórico de las columnas numéricas. No predice el futuro con certeza: ayuda a ver rangos probables, extremos y riesgos que no se ven en una tabla simple.");
            ui.add_space(10.0);

            ui.columns(4, |cols| {
                ui_kpi(&mut cols[0], "Simulaciones", &mc.simulaciones.to_string());
                ui_kpi(&mut cols[1], "Tamaño lote", &mc.tamano_lote.to_string());
                ui_kpi(&mut cols[2], "Columnas", &mc.columnas.len().to_string());
                ui_kpi(&mut cols[3], "Riesgos", &mc.riesgos.len().to_string());
            });

            ui.add_space(10.0);
            egui::Frame::new()
                .fill(W4)
                .stroke(egui::Stroke::new(1.0, BORDE))
                .corner_radius(8.0)
                .inner_margin(10.0)
                .show(ui, |ui| {
                    ui.strong("Lectura automática");
                    ui.label(&mc.conclusion);
                    ui.add_space(4.0);
                    ui.label("P5 = escenario bajo/optimista, P50 = escenario típico, P95 = escenario alto/pesimista. La interpretación depende del negocio: para costos puede ser riesgo; para producción puede ser oportunidad.");
                });

            ui.add_space(14.0);
            let datos: Vec<(String, f64)> = mc.columnas.iter()
                .map(|c| (abreviar(&c.columna, 30), c.promedio_sim_p95 - c.promedio_sim_p5))
                .collect();
            draw_horizontal_bar_chart(ui, "Amplitud del rango simulado por promedio", &datos, 12, datos.iter().map(|(_, v)| *v).fold(0.0_f64, f64::max).max(1.0));

            ui.add_space(14.0);
            ui.heading("📊 Resumen por columna simulada");
            egui::Grid::new("tabla_montecarlo_resumen")
                .striped(true)
                .min_col_width(105.0)
                .show(ui, |ui| {
                    ui.strong("Columna");
                    ui.strong("Prom. hist.");
                    ui.strong("P5 prom.");
                    ui.strong("P50 prom.");
                    ui.strong("P95 prom.");
                    ui.strong("P5 total lote");
                    ui.strong("P50 total lote");
                    ui.strong("P95 total lote");
                    ui.strong("Lectura");
                    ui.end_row();

                    for c in mc.columnas.iter().take(100) {
                        ui.label(&c.columna);
                        ui.label(format!("{:.4}", c.promedio_historico));
                        ui.label(format!("{:.4}", c.promedio_sim_p5));
                        ui.label(format!("{:.4}", c.promedio_sim_p50));
                        ui.label(format!("{:.4}", c.promedio_sim_p95));
                        ui.label(format!("{:.2}", c.total_sim_p5));
                        ui.label(format!("{:.2}", c.total_sim_p50));
                        ui.label(format!("{:.2}", c.total_sim_p95));
                        ui.label(&c.lectura);
                        ui.end_row();
                    }
                });

            ui.add_space(14.0);
            ui.heading("🚦 Riesgos y señales para revisar");
            if mc.riesgos.is_empty() {
                ui.label("No se detectaron señales fuertes de riesgo estadístico en las columnas simuladas.");
            } else {
                egui::Grid::new("tabla_montecarlo_riesgos")
                    .striped(true)
                    .min_col_width(120.0)
                    .show(ui, |ui| {
                        ui.strong("Severidad");
                        ui.strong("Columna");
                        ui.strong("Indicador");
                        ui.strong("Valor");
                        ui.strong("Lectura");
                        ui.end_row();

                        for r in mc.riesgos.iter().take(100) {
                            let color = if r.severidad == "alta" { egui::Color32::DARK_RED } else if r.severidad == "media" { A3 } else { W3 };
                            ui.colored_label(color, &r.severidad);
                            ui.label(&r.columna);
                            ui.label(&r.indicador);
                            ui.label(format!("{:.4}", r.valor));
                            ui.label(&r.lectura);
                            ui.end_row();
                        }
                    });
            }

            ui.add_space(14.0);
            ui.heading("🎯 Escenarios listos para explicar");
            egui::Grid::new("tabla_montecarlo_escenarios")
                .striped(true)
                .min_col_width(130.0)
                .show(ui, |ui| {
                    ui.strong("Columna");
                    ui.strong("Escenario");
                    ui.strong("Promedio");
                    ui.strong("Total lote");
                    ui.strong("Interpretación");
                    ui.end_row();

                    for e in mc.escenarios.iter().take(150) {
                        ui.label(&e.columna);
                        ui.label(&e.escenario);
                        ui.label(format!("{:.4}", e.promedio_estimado));
                        ui.label(format!("{:.2}", e.total_lote_estimado));
                        ui.label(&e.interpretacion);
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
    aplicar_feature_selection: bool,
    min_score_feature: f64,
    carpeta_salida_base: Option<PathBuf>,
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

    let carpeta_salida = crear_carpeta_salida(path, carpeta_salida_base.as_deref())?;

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

    set_progress(&proc, 0.52, "Aplicando Feature Selection para reducir columnas útiles...")?;

    let feature_selection_pre = calcular_feature_selection(&perfil, &[], aplicar_feature_selection, min_score_feature, &[]);
    let columnas_recomendadas_pre: HashSet<String> = feature_selection_pre
        .columnas
        .iter()
        .filter(|x| x.decision == "Usar" || x.decision == "Revisar")
        .map(|x| x.columna.clone())
        .collect();

    let mut cols_num_analisis: Vec<(String, Vec<Option<f64>>)> = if aplicar_feature_selection {
        cols_num
            .iter()
            .filter(|(nombre, _)| columnas_recomendadas_pre.contains(nombre))
            .cloned()
            .collect()
    } else {
        cols_num.clone()
    };

    if cols_num_analisis.len() < 2 {
        cols_num_analisis = cols_num.clone();
    }

    let columnas_usadas_analisis: Vec<String> = cols_num_analisis.iter().map(|(n, _)| n.clone()).collect();

    set_progress(
        &proc,
        0.55,
        &format!(
            "Calculando correlaciones con {} columnas numéricas seleccionadas...",
            cols_num_analisis.len()
        ),
    )?;

    let correlaciones = calcular_correlaciones(&cols_num_analisis, umbral);

    set_progress(
        &proc,
        0.75,
        &format!("{0} pares fuertes", correlaciones.len()),
    )?;

    set_progress(&proc, 0.80, "Agrupando columnas...")?;

    let grupos = agrupar_columnas(&correlaciones, umbral, min_grupo);

    set_progress(&proc, 0.87, &format!("{0} grupos", grupos.len()))?;

    set_progress(&proc, 0.86, "Ejecutando KMeans para detectar registros raros...")?;

    let kmeans = calcular_kmeans(&perfil, &cols_num_analisis, filas);

    set_progress(&proc, 0.89, "Ejecutando Monte Carlo para estimar escenarios...")?;

    let montecarlo = calcular_montecarlo(&perfil, &cols_num_analisis, &target_col, filas);

    set_progress(&proc, 0.92, "Generando insights y recomendaciones...")?;

    let feature_selection = calcular_feature_selection(
        &perfil,
        &correlaciones,
        aplicar_feature_selection,
        min_score_feature,
        &columnas_usadas_analisis,
    );

    let mut insights = generar_insights(&perfil, &correlaciones, &grupos, kmeans.as_ref());
    agregar_insights_montecarlo(&mut insights, montecarlo.as_ref());
    agregar_insights_feature_selection(&mut insights, Some(&feature_selection));
    let recomendaciones = recomendar_graficos(&perfil, &correlaciones);
    let ml = calcular_ml_columnas(&perfil, &cols_num_analisis, &target_col);

    set_progress(&proc, 0.95, "Escribiendo CSVs...")?;

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
    escribir_feature_selection_csv(&carpeta_salida.join("feature_selection.csv"), Some(&feature_selection))?;
    escribir_kmeans_resumen_csv(&carpeta_salida.join("kmeans_resumen.csv"), kmeans.as_ref())?;
    escribir_kmeans_outliers_csv(&carpeta_salida.join("kmeans_outliers.csv"), kmeans.as_ref())?;
    escribir_montecarlo_resumen_csv(&carpeta_salida.join("montecarlo_resumen.csv"), montecarlo.as_ref())?;
    escribir_montecarlo_escenarios_csv(&carpeta_salida.join("montecarlo_escenarios.csv"), montecarlo.as_ref())?;
    escribir_montecarlo_riesgos_csv(&carpeta_salida.join("montecarlo_riesgos.csv"), montecarlo.as_ref())?;
    escribir_reporte_html(
        &carpeta_salida.join("reporte_analitico.html"),
        filas,
        columnas,
        &perfil,
        &correlaciones,
        &grupos,
        &insights,
        &recomendaciones,
        &ml,
        kmeans.as_ref(),
        montecarlo.as_ref(),
        Some(&feature_selection),
    )?;

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
        kmeans,
        montecarlo,
        feature_selection: Some(feature_selection),
    })
}

fn set_progress(proc: &Arc<Mutex<ProcState>>, progress: f32, mensaje: &str) -> Result<()> {
    let mut p = proc.lock().unwrap();
    p.progress = progress;
    p.mensaje = mensaje.to_string();
    Ok(())
}

fn crear_carpeta_salida(path: &Path, carpeta_base: Option<&Path>) -> Result<PathBuf> {
    let base = carpeta_base
        .or_else(|| path.parent())
        .unwrap_or_else(|| Path::new("."));
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


// ─── Feature Selection ─────────────────────────────────────────────────────

fn calcular_feature_selection(
    perfil: &[PerfilColumna],
    correlaciones: &[Correlacion],
    aplicado_a_analisis: bool,
    min_score: f64,
    columnas_usadas_analisis: &[String],
) -> FeatureSelectionResultado {
    let umbral = min_score.clamp(0.0, 1.0);

    let base_scores: HashMap<String, f64> = perfil
        .iter()
        .map(|p| {
            let (score, _, _, _) = evaluar_columna_feature(p);
            (p.columna.clone(), score)
        })
        .collect();

    let mut corr_max: HashMap<String, (f64, String)> = HashMap::new();
    let mut redundante_con: HashMap<String, (String, f64)> = HashMap::new();

    for c in correlaciones {
        for (a, b) in [(&c.columna_1, &c.columna_2), (&c.columna_2, &c.columna_1)] {
            let entry = corr_max.entry(a.clone()).or_insert((0.0, String::new()));
            if c.correlacion_absoluta > entry.0 {
                *entry = (c.correlacion_absoluta, b.clone());
            }
        }

        if c.correlacion_absoluta >= 0.98 {
            let s1 = base_scores.get(&c.columna_1).copied().unwrap_or(0.0);
            let s2 = base_scores.get(&c.columna_2).copied().unwrap_or(0.0);

            if s1 >= s2 {
                redundante_con.entry(c.columna_2.clone()).or_insert((c.columna_1.clone(), c.correlacion_absoluta));
            } else {
                redundante_con.entry(c.columna_1.clone()).or_insert((c.columna_2.clone(), c.correlacion_absoluta));
            }
        }
    }

    let mut columnas = Vec::new();

    for p in perfil {
        let (mut score, mut motivo, mut accion, mut riesgo) = evaluar_columna_feature(p);
        let (correlacion_maxima, correlacion_con) = corr_max
            .get(&p.columna)
            .map(|(v, c)| (Some(*v), c.clone()))
            .unwrap_or((None, String::new()));

        if let Some((otra, corr)) = redundante_con.get(&p.columna) {
            score = score.min((umbral - 0.01).max(0.0));
            riesgo = "medio".into();
            motivo = format!("{motivo}; redundante con {otra} (corr={corr:.4})");
            accion = format!("Puede descartarse para análisis porque otra columna conserva casi la misma información: {otra}");
        }

        let decision = if score >= umbral {
            "Usar"
        } else if score >= (umbral * 0.75).max(0.40) {
            "Revisar"
        } else {
            "Descartar"
        };

        columnas.push(FeatureSelectionColumna {
            columna: p.columna.clone(),
            decision: decision.into(),
            score,
            tipo_analitico: p.tipo_analitico.clone(),
            porcentaje_nulos: p.porcentaje_nulos,
            porcentaje_unicos: p.porcentaje_unicos,
            correlacion_maxima,
            correlacion_con,
            motivo,
            accion,
            riesgo,
        });
    }

    columnas.sort_by(|a, b| {
        let orden = |d: &str| match d {
            "Usar" => 0,
            "Revisar" => 1,
            _ => 2,
        };

        orden(&a.decision)
            .cmp(&orden(&b.decision))
            .then_with(|| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal))
    });

    let columnas_recomendadas = columnas.iter().filter(|x| x.decision == "Usar").count();
    let columnas_revision = columnas.iter().filter(|x| x.decision == "Revisar").count();
    let columnas_descartadas = columnas.iter().filter(|x| x.decision == "Descartar").count();

    let mut usadas = columnas_usadas_analisis.to_vec();
    usadas.sort();
    usadas.dedup();

    let conclusion = if aplicado_a_analisis {
        format!(
            "Feature Selection recomendó usar {} de {} columnas, revisar {} y descartar {}. Los análisis pesados trabajaron con {} columnas numéricas seleccionadas para reducir ruido, memoria y tiempo.",
            columnas_recomendadas,
            perfil.len(),
            columnas_revision,
            columnas_descartadas,
            usadas.len()
        )
    } else {
        format!(
            "Feature Selection recomendó usar {} de {} columnas, revisar {} y descartar {}. No se aplicó como filtro, por lo que sirve como guía de limpieza y priorización.",
            columnas_recomendadas,
            perfil.len(),
            columnas_revision,
            columnas_descartadas
        )
    };

    FeatureSelectionResultado {
        columnas_originales: perfil.len(),
        columnas_recomendadas,
        columnas_descartadas,
        columnas_revision,
        aplicado_a_analisis,
        umbral_score: umbral,
        columnas_usadas_analisis: usadas,
        columnas,
        conclusion,
    }
}

fn evaluar_columna_feature(p: &PerfilColumna) -> (f64, String, String, String) {
    let mut score = 1.0_f64;
    let mut motivos: Vec<String> = Vec::new();
    let mut acciones: Vec<String> = Vec::new();
    let mut riesgo = "bajo".to_string();

    let nombre = p.columna.to_lowercase();
    let pct_unicos = p.porcentaje_unicos.unwrap_or(0.0);
    let unicos = p.valores_unicos.unwrap_or(0);

    if p.total_filas == 0 || p.no_nulos == 0 {
        return (
            0.0,
            "columna vacía o sin datos válidos".into(),
            "Descartar para análisis estadístico".into(),
            "alto".into(),
        );
    }

    if p.porcentaje_nulos >= 95.0 {
        score = 0.0;
        motivos.push("tiene 95% o más de nulos".into());
        acciones.push("Descartar o corregir fuente de datos".into());
        riesgo = "alto".into();
    } else if p.porcentaje_nulos >= 70.0 {
        score -= 0.55;
        motivos.push("tiene demasiados nulos".into());
        acciones.push("Revisar si la columna es obligatoria o si debe imputarse".into());
        riesgo = "alto".into();
    } else if p.porcentaje_nulos >= 40.0 {
        score -= 0.25;
        motivos.push("tiene nulos importantes".into());
        acciones.push("Puede usarse, pero conviene revisar calidad".into());
        riesgo = "medio".into();
    } else {
        motivos.push("tiene completitud aceptable".into());
    }

    if unicos <= 1 {
        score = 0.0;
        motivos.push("es constante o no varía".into());
        acciones.push("Descartar porque no ayuda a diferenciar registros".into());
        riesgo = "alto".into();
    } else if unicos <= 2 && p.tipo_analitico == "numérica" {
        score -= 0.25;
        motivos.push("tiene muy pocos valores distintos".into());
        acciones.push("Revisar si realmente representa una variable numérica o una categoría".into());
        riesgo = if riesgo == "bajo" { "medio".into() } else { riesgo };
    }

    let parece_id_por_nombre = nombre.contains("id")
        || nombre.contains("codigo")
        || nombre.contains("código")
        || nombre.contains("cod_")
        || nombre.starts_with("cod")
        || nombre.contains("cedula")
        || nombre.contains("cédula")
        || nombre.contains("dni")
        || nombre.contains("ruc")
        || nombre.contains("uuid")
        || nombre.contains("tramite")
        || nombre.contains("trámite")
        || nombre.contains("secuencia")
        || nombre.contains("numero")
        || nombre.contains("número");

    if parece_id_por_nombre && pct_unicos >= 70.0 {
        score -= 0.55;
        motivos.push("parece identificador por nombre y alta unicidad".into());
        acciones.push("No usar para KMeans, correlaciones ni ML predictivo; conservar solo como referencia".into());
        riesgo = "medio".into();
    } else if p.tipo_analitico == "texto/id" && pct_unicos >= 80.0 {
        score -= 0.45;
        motivos.push("parece texto libre o identificador".into());
        acciones.push("Evitar en análisis numérico; usar solo para búsqueda o trazabilidad".into());
        riesgo = if riesgo == "bajo" { "medio".into() } else { riesgo };
    } else if p.tipo_analitico == "texto/id" {
        score -= 0.20;
        motivos.push("es texto de alta cardinalidad".into());
        acciones.push("Usar solo si luego se transforma con técnicas de texto".into());
    }

    if p.tipo_analitico == "categórica" && (unicos > 1_000 || pct_unicos >= 50.0) {
        score -= 0.30;
        motivos.push("tiene demasiadas categorías".into());
        acciones.push("Agrupar categorías raras o usar top N antes de graficar".into());
        riesgo = if riesgo == "bajo" { "medio".into() } else { riesgo };
    }

    if p.tipo_analitico == "numérica" {
        match p.desviacion_estandar {
            Some(std) if std.is_finite() && std > 0.0 => {
                motivos.push("tiene variabilidad numérica útil".into());
            }
            _ => {
                score = 0.0;
                motivos.push("no tiene variabilidad numérica útil".into());
                acciones.push("Descartar para correlaciones, KMeans y Monte Carlo".into());
                riesgo = "alto".into();
            }
        }

        if p.outliers_iqr_muestra.unwrap_or(0) > 0 {
            score -= 0.03;
            motivos.push("presenta posibles outliers".into());
            acciones.push("Revisar extremos antes de tomar decisiones".into());
        }
    }

    if p.tipo_analitico == "temporal" {
        score -= 0.05;
        motivos.push("es temporal".into());
        acciones.push("Puede servir para tendencias; convertir a año, mes, día o antigüedad si se usará en ML".into());
    }

    if acciones.is_empty() {
        acciones.push("Usar como candidata principal para análisis".into());
    }

    score = score.clamp(0.0, 1.0);

    (
        score,
        motivos.join("; "),
        acciones.join("; "),
        riesgo,
    )
}

fn agregar_insights_feature_selection(insights: &mut Vec<InsightAuto>, fs: Option<&FeatureSelectionResultado>) {
    let Some(fs) = fs else { return; };

    let severidad = if fs.columnas_descartadas > fs.columnas_recomendadas {
        "media"
    } else {
        "baja"
    };

    insights.push(InsightAuto {
        categoria: "feature_selection".into(),
        severidad: severidad.into(),
        columna: "todas".into(),
        mensaje: fs.conclusion.clone(),
    });

    for x in fs.columnas.iter().filter(|x| x.decision == "Descartar").take(5) {
        insights.push(InsightAuto {
            categoria: "feature_selection".into(),
            severidad: if x.riesgo == "alto" { "alta".into() } else { "media".into() },
            columna: x.columna.clone(),
            mensaje: format!("Columna sugerida para descartar: {}. Motivo: {}", x.accion, x.motivo),
        });
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
    kmeans: Option<&KMeansResultado>,
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

    if let Some(km) = kmeans {
        if !km.outliers.is_empty() {
            out.push(InsightAuto {
                categoria: "kmeans".into(),
                severidad: if km.outliers.len() >= 20 { "alta".into() } else { "media".into() },
                columna: "registros raros".into(),
                mensaje: format!(
                    "KMeans detectó {} registros fuera del comportamiento normal de la muestra. Conviene revisar kmeans_outliers.csv.",
                    km.outliers.len()
                ),
            });
        }

        for c in km.clusters.iter().filter(|c| c.porcentaje <= 3.0).take(10) {
            out.push(InsightAuto {
                categoria: "segmento pequeño".into(),
                severidad: "media".into(),
                columna: format!("Cluster {}", c.cluster),
                mensaje: format!(
                    "Solo contiene {:.2}% de la muestra. Puede representar un grupo atípico, una regla de negocio diferente o datos mal cargados.",
                    c.porcentaje
                ),
            });
        }
    }

    out.sort_by(|a, b| prioridad_val(&b.severidad).cmp(&prioridad_val(&a.severidad)));
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


// ─── KMeans explicable para detectar patrones no evidentes ──────────────────

fn calcular_kmeans(
    perfil: &[PerfilColumna],
    cols_num: &[(String, Vec<Option<f64>>) ],
    filas_total: usize,
) -> Option<KMeansResultado> {
    let seleccion = seleccionar_features_kmeans(perfil, cols_num);

    if seleccion.len() < 2 {
        return None;
    }

    let min_len = seleccion.iter().map(|(_, v)| v.len()).min().unwrap_or(0);

    if min_len < 30 {
        return None;
    }

    let step = if min_len > MAX_FILAS_KMEANS {
        ((min_len as f64 / MAX_FILAS_KMEANS as f64).ceil() as usize).max(1)
    } else {
        1
    };

    let mut medias = Vec::new();
    let mut desvios = Vec::new();

    for (_, v) in &seleccion {
        let vals: Vec<f64> = v.iter()
            .filter_map(|x| *x)
            .filter(|x| x.is_finite())
            .collect();

        if vals.len() < 10 {
            return None;
        }

        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        let std = var.sqrt();

        if !std.is_finite() || std <= 0.0 {
            return None;
        }

        medias.push(mean);
        desvios.push(std);
    }

    let mut matriz: Vec<Vec<f64>> = Vec::new();
    let mut indices_muestra: Vec<usize> = Vec::new();

    for row_idx in (0..min_len).step_by(step) {
        let mut fila = Vec::with_capacity(seleccion.len());
        let mut validos = 0usize;

        for (j, (_, col)) in seleccion.iter().enumerate() {
            let z = match col.get(row_idx).and_then(|x| *x) {
                Some(v) if v.is_finite() => {
                    validos += 1;
                    ((v - medias[j]) / desvios[j]).clamp(-8.0, 8.0)
                }
                _ => 0.0, // nulo = valor promedio estandarizado
            };

            fila.push(z);
        }

        if validos >= 2 {
            matriz.push(fila);
            indices_muestra.push(row_idx);
        }
    }

    let n = matriz.len();
    let d = seleccion.len();

    if n < 30 || d < 2 {
        return None;
    }

    let k = elegir_k_auto(n);

    if k < 2 || k >= n {
        return None;
    }

    let mut centroides: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            let idx = ((i as f64 + 0.5) * n as f64 / k as f64).floor() as usize;
            matriz[idx.min(n - 1)].clone()
        })
        .collect();

    let mut asignaciones = vec![0usize; n];

    for _ in 0..35 {
        let mut cambio = false;

        for (i, fila) in matriz.iter().enumerate() {
            let mut mejor = 0usize;
            let mut mejor_dist = f64::INFINITY;

            for (cidx, c) in centroides.iter().enumerate() {
                let dist = distancia2(fila, c);

                if dist < mejor_dist {
                    mejor_dist = dist;
                    mejor = cidx;
                }
            }

            if asignaciones[i] != mejor {
                asignaciones[i] = mejor;
                cambio = true;
            }
        }

        let mut nuevos = vec![vec![0.0; d]; k];
        let mut counts = vec![0usize; k];

        for (fila, &cluster) in matriz.iter().zip(asignaciones.iter()) {
            counts[cluster] += 1;

            for j in 0..d {
                nuevos[cluster][j] += fila[j];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d {
                    nuevos[c][j] /= counts[c] as f64;
                }
            } else {
                let idx = ((c as f64 + 0.5) * n as f64 / k as f64).floor() as usize;
                nuevos[c] = matriz[idx.min(n - 1)].clone();
            }
        }

        centroides = nuevos;

        if !cambio {
            break;
        }
    }

    let mut distancias = Vec::with_capacity(n);
    let mut dist_por_cluster = vec![0.0; k];
    let mut counts = vec![0usize; k];

    for (i, fila) in matriz.iter().enumerate() {
        let cluster = asignaciones[i];
        let dist = distancia2(fila, &centroides[cluster]).sqrt();
        distancias.push(dist);
        dist_por_cluster[cluster] += dist;
        counts[cluster] += 1;
    }

    let distancia_promedio = distancias.iter().sum::<f64>() / distancias.len() as f64;
    let var_dist = distancias.iter().map(|x| (x - distancia_promedio).powi(2)).sum::<f64>() / distancias.len() as f64;
    let std_dist = var_dist.sqrt();

    let mut dist_sort = distancias.clone();
    dist_sort.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95 = percentil_sorted(&dist_sort, 0.95);
    let p99 = percentil_sorted(&dist_sort, 0.99);

    let mut clusters = Vec::new();

    for c in 0..k {
        let cantidad = counts[c];
        let porcentaje = if n > 0 { cantidad as f64 * 100.0 / n as f64 } else { 0.0 };
        let dist_prom = if cantidad > 0 { dist_por_cluster[c] / cantidad as f64 } else { 0.0 };
        let lectura = if porcentaje <= 3.0 {
            "segmento pequeño: revisar porque puede ser comportamiento especial o carga rara".to_string()
        } else if dist_prom >= p95 {
            "cluster disperso: sus filas son menos homogéneas".to_string()
        } else {
            "cluster estable".to_string()
        };

        clusters.push(KMeansCluster {
            cluster: c,
            cantidad,
            porcentaje,
            distancia_promedio: dist_prom,
            lectura,
        });
    }

    clusters.sort_by(|a, b| b.cantidad.cmp(&a.cantidad));

    let mut candidatos: Vec<(usize, f64)> = distancias.iter()
        .enumerate()
        .filter_map(|(i, d)| {
            let z = if std_dist > 0.0 { (*d - distancia_promedio) / std_dist } else { 0.0 };
            if *d >= p95 || z >= 2.5 { Some((i, *d)) } else { None }
        })
        .collect();

    candidatos.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let factor_aprox = if filas_total > min_len && min_len > 0 {
        (filas_total as f64 / min_len as f64).ceil() as usize
    } else {
        1
    };

    let features: Vec<String> = seleccion.iter().map(|(n, _)| n.clone()).collect();

    let mut outliers = Vec::new();

    for (rank, (idx, dist)) in candidatos.into_iter().take(200).enumerate() {
        let z = if std_dist > 0.0 { (dist - distancia_promedio) / std_dist } else { 0.0 };
        let severidad = if dist >= p99 || z >= 3.0 { "alta" } else { "media" };
        let fila_aprox = indices_muestra.get(idx).copied().unwrap_or(idx).saturating_mul(factor_aprox) + 1;
        let valores_principales = explicar_fila_rara(&features, &matriz[idx], &centroides[asignaciones[idx]]);

        outliers.push(KMeansOutlier {
            ranking: rank + 1,
            fila_muestra: indices_muestra.get(idx).copied().unwrap_or(idx) + 1,
            fila_aproximada: fila_aprox,
            cluster: asignaciones[idx],
            distancia: dist,
            z_score: z,
            severidad: severidad.into(),
            razon: if severidad == "alta" {
                "muy alejada del centro de su cluster".into()
            } else {
                "más alejada que la mayoría de registros".into()
            },
            valores_principales,
        });
    }

    let conclusion = if outliers.is_empty() {
        format!(
            "Se formaron {k} grupos con {} variables numéricas. No aparecieron registros extremadamente alejados; el comportamiento general luce consistente.",
            features.len()
        )
    } else {
        format!(
            "Se formaron {k} grupos con {} variables numéricas y se detectaron {} registros raros en la muestra. Estos casos no deben asumirse como error, pero sí conviene revisarlos primero.",
            features.len(),
            outliers.len()
        )
    };

    Some(KMeansResultado {
        features,
        k,
        filas_muestra: n,
        clusters,
        outliers,
        distancia_promedio,
        distancia_p95: p95,
        distancia_p99: p99,
        conclusion,
    })
}

fn seleccionar_features_kmeans<'a>(
    perfil: &'a [PerfilColumna],
    cols_num: &'a [(String, Vec<Option<f64>>) ],
) -> Vec<(String, &'a Vec<Option<f64>>)> {
    let mut candidatos: Vec<(f64, String, &'a Vec<Option<f64>>)> = Vec::new();

    for p in perfil {
        if p.tipo_analitico != "numérica" {
            continue;
        }

        if p.porcentaje_nulos > 45.0 {
            continue;
        }

        if p.desviacion_estandar.unwrap_or(0.0) <= 0.0 {
            continue;
        }

        if p.valores_unicos.unwrap_or(0) <= 2 {
            continue;
        }

        let parece_id = p.porcentaje_unicos.unwrap_or(0.0) >= 95.0 && p.total_filas >= 500;

        if parece_id {
            continue;
        }

        if let Some((_, col)) = cols_num.iter().find(|(n, _)| n == &p.columna) {
            let completitud = 1.0 - (p.porcentaje_nulos / 100.0).clamp(0.0, 1.0);
            let variacion = p.desviacion_estandar.unwrap_or(0.0).abs().ln_1p().clamp(0.0, 10.0) / 10.0;
            let outlier_bonus = (p.outliers_iqr_muestra.unwrap_or(0) as f64).ln_1p().clamp(0.0, 8.0) / 8.0;
            let score = 0.65 * completitud + 0.20 * variacion + 0.15 * outlier_bonus;
            candidatos.push((score, p.columna.clone(), col));
        }
    }

    candidatos.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    candidatos.into_iter()
        .take(MAX_COLUMNAS_KMEANS)
        .map(|(_, n, c)| (n, c))
        .collect()
}

fn elegir_k_auto(n: usize) -> usize {
    let base = if n < 300 {
        3
    } else if n < 3_000 {
        4
    } else if n < 20_000 {
        5
    } else {
        6
    };

    base.min((n / 20).max(2)).max(2)
}

fn distancia2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn explicar_fila_rara(features: &[String], fila: &[f64], centroide: &[f64]) -> String {
    let mut partes: Vec<(f64, String)> = fila.iter()
        .zip(centroide.iter())
        .zip(features.iter())
        .map(|((v, c), nombre)| ((v - c).abs(), format!("{}={:.2}σ", nombre, v)))
        .collect();

    partes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    partes.into_iter()
        .take(4)
        .map(|(_, txt)| txt)
        .collect::<Vec<_>>()
        .join("; ")
}


// ─── Monte Carlo automático para escenarios no evidentes ───────────────────

fn calcular_montecarlo(
    perfil: &[PerfilColumna],
    cols_num: &[(String, Vec<Option<f64>>) ],
    target_col: &str,
    _filas_total: usize,
) -> Option<MonteCarloResultado> {
    let seleccion = seleccionar_columnas_montecarlo(perfil, cols_num, target_col);

    if seleccion.is_empty() {
        return None;
    }

    let mut columnas = Vec::new();
    let mut escenarios = Vec::new();
    let mut riesgos = Vec::new();

    for (idx_col, (nombre, datos, perfil_col)) in seleccion.iter().enumerate() {
        let mut vals: Vec<f64> = datos.iter()
            .filter_map(|x| *x)
            .filter(|x| x.is_finite())
            .collect();

        if vals.len() < 30 {
            continue;
        }

        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let promedio = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|x| (x - promedio).powi(2)).sum::<f64>() / vals.len() as f64;
        let desv = var.sqrt();

        if !promedio.is_finite() || !desv.is_finite() || desv <= 0.0 {
            continue;
        }

        let min_hist = *vals.first().unwrap_or(&0.0);
        let max_hist = *vals.last().unwrap_or(&0.0);
        let p5_hist = percentil_sorted(&vals, 0.05);
        let p50_hist = percentil_sorted(&vals, 0.50);
        let p95_hist = percentil_sorted(&vals, 0.95);

        let tamano_lote = MONTECARLO_TAMANO_LOTE.min(vals.len()).max(20);
        let mut rng = Lcg::new(0x9E37_79B9_7F4A_7C15 ^ hash_nombre(nombre) ^ ((idx_col as u64) << 32));
        let mut promedios_sim = Vec::with_capacity(MONTECARLO_SIMULACIONES);
        let mut supera_p95_hist = 0usize;
        let mut bajo_p5_hist = 0usize;

        for _ in 0..MONTECARLO_SIMULACIONES {
            let mut suma = 0.0;

            for _ in 0..tamano_lote {
                let idx = rng.gen_range(vals.len());
                let v = vals[idx];
                suma += v;

                if v > p95_hist {
                    supera_p95_hist += 1;
                }

                if v < p5_hist {
                    bajo_p5_hist += 1;
                }
            }

            promedios_sim.push(suma / tamano_lote as f64);
        }

        promedios_sim.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let prom_p5 = percentil_sorted(&promedios_sim, 0.05);
        let prom_p50 = percentil_sorted(&promedios_sim, 0.50);
        let prom_p95 = percentil_sorted(&promedios_sim, 0.95);

        let total_p5 = prom_p5 * tamano_lote as f64;
        let total_p50 = prom_p50 * tamano_lote as f64;
        let total_p95 = prom_p95 * tamano_lote as f64;
        let total_sorteos = (MONTECARLO_SIMULACIONES * tamano_lote).max(1) as f64;
        let prob_superar_p95 = supera_p95_hist as f64 * 100.0 / total_sorteos;
        let prob_bajo_p5 = bajo_p5_hist as f64 * 100.0 / total_sorteos;

        let rango_relativo = if prom_p50.abs() > 1e-9 {
            ((prom_p95 - prom_p5).abs() / prom_p50.abs()).abs()
        } else {
            (prom_p95 - prom_p5).abs()
        };

        let cv = if promedio.abs() > 1e-9 { (desv / promedio.abs()).abs() } else { desv.abs() };

        let lectura = if rango_relativo >= 1.0 {
            "rango simulado amplio: el comportamiento puede cambiar bastante entre escenarios".to_string()
        } else if cv >= 0.75 {
            "alta variabilidad histórica: conviene revisar extremos antes de tomar decisiones".to_string()
        } else if perfil_col.outliers_iqr_muestra.unwrap_or(0) > 0 {
            "tiene valores extremos: revisar boxplot y escenarios altos".to_string()
        } else {
            "comportamiento relativamente estable en la simulación".to_string()
        };

        escenarios.push(MonteCarloEscenario {
            columna: nombre.clone(),
            escenario: "Bajo / P5".into(),
            promedio_estimado: prom_p5,
            total_lote_estimado: total_p5,
            interpretacion: "Escenario bajo. Puede ser favorable o desfavorable según el significado de la columna.".into(),
        });
        escenarios.push(MonteCarloEscenario {
            columna: nombre.clone(),
            escenario: "Probable / P50".into(),
            promedio_estimado: prom_p50,
            total_lote_estimado: total_p50,
            interpretacion: "Escenario central. Es el valor más razonable para empezar una explicación ejecutiva.".into(),
        });
        escenarios.push(MonteCarloEscenario {
            columna: nombre.clone(),
            escenario: "Alto / P95".into(),
            promedio_estimado: prom_p95,
            total_lote_estimado: total_p95,
            interpretacion: "Escenario alto. Para costos, tiempos o errores puede representar riesgo; para producción o ingresos puede representar oportunidad.".into(),
        });

        if cv >= 1.0 {
            riesgos.push(MonteCarloRiesgo {
                columna: nombre.clone(),
                severidad: "alta".into(),
                indicador: "coeficiente_variacion".into(),
                valor: cv,
                lectura: "La desviación es mayor o similar al promedio. Hay mucha dispersión y puede ocultar casos extremos.".into(),
            });
        } else if cv >= 0.50 {
            riesgos.push(MonteCarloRiesgo {
                columna: nombre.clone(),
                severidad: "media".into(),
                indicador: "coeficiente_variacion".into(),
                valor: cv,
                lectura: "La columna varía bastante. Conviene revisar segmentación por grupos o períodos.".into(),
            });
        }

        if rango_relativo >= 1.0 {
            riesgos.push(MonteCarloRiesgo {
                columna: nombre.clone(),
                severidad: "alta".into(),
                indicador: "rango_simulado_relativo".into(),
                valor: rango_relativo,
                lectura: "Entre el escenario bajo y alto hay una diferencia muy grande. La columna puede cambiar la lectura del negocio.".into(),
            });
        } else if rango_relativo >= 0.35 {
            riesgos.push(MonteCarloRiesgo {
                columna: nombre.clone(),
                severidad: "media".into(),
                indicador: "rango_simulado_relativo".into(),
                valor: rango_relativo,
                lectura: "El rango de escenarios es moderado. Sirve para explicar incertidumbre sin entrar en matemática compleja.".into(),
            });
        }

        if let Some(asim) = perfil_col.asimetria {
            if asim.abs() >= 2.0 {
                riesgos.push(MonteCarloRiesgo {
                    columna: nombre.clone(),
                    severidad: "media".into(),
                    indicador: "asimetria".into(),
                    valor: asim,
                    lectura: "La distribución está cargada hacia un lado. El promedio puede engañar; revisar mediana y percentiles.".into(),
                });
            }
        }

        if let Some(out) = perfil_col.outliers_iqr_muestra {
            let denom = perfil_col.no_nulos.min(MAX_FILAS_MUESTRA).max(1) as f64;
            let pct_out = out as f64 * 100.0 / denom;

            if pct_out >= 5.0 {
                riesgos.push(MonteCarloRiesgo {
                    columna: nombre.clone(),
                    severidad: "media".into(),
                    indicador: "outliers_iqr_pct".into(),
                    valor: pct_out,
                    lectura: "Hay varios valores extremos. En escenarios altos pueden pesar mucho.".into(),
                });
            }
        }

        columnas.push(MonteCarloColumna {
            columna: nombre.clone(),
            muestras_validas: vals.len(),
            promedio_historico: promedio,
            desviacion_historica: desv,
            minimo_historico: min_hist,
            p5_historico: p5_hist,
            p50_historico: p50_hist,
            p95_historico: p95_hist,
            maximo_historico: max_hist,
            promedio_sim_p5: prom_p5,
            promedio_sim_p50: prom_p50,
            promedio_sim_p95: prom_p95,
            total_sim_p5: total_p5,
            total_sim_p50: total_p50,
            total_sim_p95: total_p95,
            prob_superar_p95,
            prob_bajo_p5,
            lectura,
        });
    }

    if columnas.is_empty() {
        return None;
    }

    riesgos.sort_by(|a, b| prioridad_val(&b.severidad).cmp(&prioridad_val(&a.severidad)));

    let conclusion = if riesgos.iter().any(|r| r.severidad == "alta") {
        format!(
            "Monte Carlo simuló {} escenarios para {} columnas y encontró señales altas de variabilidad o incertidumbre. Conviene revisar primero las columnas marcadas en riesgos.",
            MONTECARLO_SIMULACIONES,
            columnas.len()
        )
    } else if !riesgos.is_empty() {
        format!(
            "Monte Carlo simuló {} escenarios para {} columnas. Hay señales moderadas que ayudan a explicar rangos probables, pero no aparecen alertas extremas.",
            MONTECARLO_SIMULACIONES,
            columnas.len()
        )
    } else {
        format!(
            "Monte Carlo simuló {} escenarios para {} columnas. Los rangos simulados lucen relativamente estables.",
            MONTECARLO_SIMULACIONES,
            columnas.len()
        )
    };

    let tamano_lote_resultado = columnas.iter()
        .map(|c| c.muestras_validas.min(MONTECARLO_TAMANO_LOTE).max(20))
        .min()
        .unwrap_or(MONTECARLO_TAMANO_LOTE);

    Some(MonteCarloResultado {
        simulaciones: MONTECARLO_SIMULACIONES,
        tamano_lote: tamano_lote_resultado,
        columnas,
        escenarios,
        riesgos,
        conclusion,
    })
}

fn seleccionar_columnas_montecarlo<'a>(
    perfil: &'a [PerfilColumna],
    cols_num: &'a [(String, Vec<Option<f64>>) ],
    target_col: &str,
) -> Vec<(String, &'a Vec<Option<f64>>, &'a PerfilColumna)> {
    let mut seleccion: Vec<(String, &'a Vec<Option<f64>>, &'a PerfilColumna)> = Vec::new();
    let target = target_col.trim();

    if !target.is_empty() {
        if let Some(p) = perfil.iter().find(|p| p.columna == target && p.tipo_analitico == "numérica") {
            if let Some((_, col)) = cols_num.iter().find(|(n, _)| n == target) {
                seleccion.push((p.columna.clone(), col, p));
            }
        }
    }

    let mut candidatos: Vec<(f64, String, &'a Vec<Option<f64>>, &'a PerfilColumna)> = Vec::new();

    for p in perfil {
        if p.tipo_analitico != "numérica" {
            continue;
        }

        if seleccion.iter().any(|(n, _, _)| n == &p.columna) {
            continue;
        }

        if p.porcentaje_nulos > 40.0 || p.desviacion_estandar.unwrap_or(0.0) <= 0.0 || p.valores_unicos.unwrap_or(0) <= 5 {
            continue;
        }

        let parece_id = p.porcentaje_unicos.unwrap_or(0.0) >= 95.0 && p.total_filas >= 500;
        let nombre_lower = p.columna.to_lowercase();
        let nombre_parece_id = nombre_lower.contains("id") || nombre_lower.contains("cedula") || nombre_lower.contains("cédula") || nombre_lower.contains("codigo") || nombre_lower.contains("código");

        if parece_id || nombre_parece_id {
            continue;
        }

        if let Some((_, col)) = cols_num.iter().find(|(n, _)| n == &p.columna) {
            let completitud = 1.0 - (p.porcentaje_nulos / 100.0).clamp(0.0, 1.0);
            let variacion = p.desviacion_estandar.unwrap_or(0.0).abs().ln_1p().clamp(0.0, 10.0) / 10.0;
            let outlier_bonus = (p.outliers_iqr_muestra.unwrap_or(0) as f64).ln_1p().clamp(0.0, 8.0) / 8.0;
            let asim_bonus = p.asimetria.unwrap_or(0.0).abs().clamp(0.0, 5.0) / 5.0;
            let score = 0.45 * completitud + 0.25 * variacion + 0.20 * outlier_bonus + 0.10 * asim_bonus;
            candidatos.push((score, p.columna.clone(), col, p));
        }
    }

    candidatos.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    for (_, nombre, col, p) in candidatos.into_iter() {
        if seleccion.len() >= MAX_COLUMNAS_MONTECARLO {
            break;
        }
        seleccion.push((nombre, col, p));
    }

    seleccion
}

fn agregar_insights_montecarlo(insights: &mut Vec<InsightAuto>, mc: Option<&MonteCarloResultado>) {
    let Some(mc) = mc else { return; };

    if !mc.riesgos.is_empty() {
        let altas = mc.riesgos.iter().filter(|r| r.severidad == "alta").count();
        let severidad = if altas > 0 { "alta" } else { "media" };
        insights.push(InsightAuto {
            categoria: "montecarlo".into(),
            severidad: severidad.into(),
            columna: "escenarios simulados".into(),
            mensaje: format!(
                "Monte Carlo encontró {} señales de incertidumbre en {} columnas. Revisar montecarlo_riesgos.csv.",
                mc.riesgos.len(),
                mc.columnas.len()
            ),
        });
    }

    for r in mc.riesgos.iter().take(8) {
        insights.push(InsightAuto {
            categoria: "riesgo".into(),
            severidad: r.severidad.clone(),
            columna: r.columna.clone(),
            mensaje: format!("{} = {:.4}. {}", r.indicador, r.valor, r.lectura),
        });
    }

    insights.sort_by(|a, b| prioridad_val(&b.severidad).cmp(&prioridad_val(&a.severidad)));
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            0
        } else {
            (self.next_u64() as usize) % upper
        }
    }
}

fn hash_nombre(s: &str) -> u64 {
    let mut h = 1469598103934665603u64;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
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


fn escribir_feature_selection_csv(path: &Path, fs: Option<&FeatureSelectionResultado>) -> Result<()> {
    let mut f = File::create(path)?;

    writeln!(
        f,
        "decision,score,columna,tipo_analitico,porcentaje_nulos,porcentaje_unicos,correlacion_maxima,correlacion_con,riesgo,motivo,accion"
    )?;

    if let Some(fs) = fs {
        for x in &fs.columnas {
            writeln!(
                f,
                "{},{:.6},{},{},{:.6},{},{},{},{},{},{}",
                csv(&x.decision),
                x.score,
                csv(&x.columna),
                csv(&x.tipo_analitico),
                x.porcentaje_nulos,
                opt_f64(x.porcentaje_unicos),
                opt_f64(x.correlacion_maxima),
                csv(&x.correlacion_con),
                csv(&x.riesgo),
                csv(&x.motivo),
                csv(&x.accion)
            )?;
        }
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

fn escribir_kmeans_resumen_csv(path: &Path, km: Option<&KMeansResultado>) -> Result<()> {
    let mut f = File::create(path)?;

    writeln!(f, "cluster,cantidad,porcentaje,distancia_promedio,lectura")?;

    if let Some(km) = km {
        for c in &km.clusters {
            writeln!(
                f,
                "{},{},{:.6},{:.6},{}",
                c.cluster,
                c.cantidad,
                c.porcentaje,
                c.distancia_promedio,
                csv(&c.lectura)
            )?;
        }
    }

    Ok(())
}

fn escribir_kmeans_outliers_csv(path: &Path, km: Option<&KMeansResultado>) -> Result<()> {
    let mut f = File::create(path)?;

    writeln!(
        f,
        "ranking,fila_muestra,fila_aproximada,cluster,distancia,z_score,severidad,razon,valores_principales"
    )?;

    if let Some(km) = km {
        for x in &km.outliers {
            writeln!(
                f,
                "{},{},{},{},{:.8},{:.6},{},{},{}",
                x.ranking,
                x.fila_muestra,
                x.fila_aproximada,
                x.cluster,
                x.distancia,
                x.z_score,
                csv(&x.severidad),
                csv(&x.razon),
                csv(&x.valores_principales)
            )?;
        }
    }

    Ok(())
}


fn escribir_montecarlo_resumen_csv(path: &Path, mc: Option<&MonteCarloResultado>) -> Result<()> {
    let mut f = File::create(path)?;

    writeln!(
        f,
        "columna,muestras_validas,promedio_historico,desviacion_historica,minimo_historico,p5_historico,p50_historico,p95_historico,maximo_historico,promedio_sim_p5,promedio_sim_p50,promedio_sim_p95,total_sim_p5,total_sim_p50,total_sim_p95,prob_valor_superar_p95_historico,prob_valor_bajo_p5_historico,lectura"
    )?;

    if let Some(mc) = mc {
        for c in &mc.columnas {
            writeln!(
                f,
                "{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.6},{:.6},{}",
                csv(&c.columna),
                c.muestras_validas,
                c.promedio_historico,
                c.desviacion_historica,
                c.minimo_historico,
                c.p5_historico,
                c.p50_historico,
                c.p95_historico,
                c.maximo_historico,
                c.promedio_sim_p5,
                c.promedio_sim_p50,
                c.promedio_sim_p95,
                c.total_sim_p5,
                c.total_sim_p50,
                c.total_sim_p95,
                c.prob_superar_p95,
                c.prob_bajo_p5,
                csv(&c.lectura)
            )?;
        }
    }

    Ok(())
}

fn escribir_montecarlo_escenarios_csv(path: &Path, mc: Option<&MonteCarloResultado>) -> Result<()> {
    let mut f = File::create(path)?;

    writeln!(f, "columna,escenario,promedio_estimado,total_lote_estimado,interpretacion")?;

    if let Some(mc) = mc {
        for e in &mc.escenarios {
            writeln!(
                f,
                "{},{},{:.8},{:.8},{}",
                csv(&e.columna),
                csv(&e.escenario),
                e.promedio_estimado,
                e.total_lote_estimado,
                csv(&e.interpretacion)
            )?;
        }
    }

    Ok(())
}

fn escribir_montecarlo_riesgos_csv(path: &Path, mc: Option<&MonteCarloResultado>) -> Result<()> {
    let mut f = File::create(path)?;

    writeln!(f, "severidad,columna,indicador,valor,lectura")?;

    if let Some(mc) = mc {
        for r in &mc.riesgos {
            writeln!(
                f,
                "{},{},{},{:.8},{}",
                csv(&r.severidad),
                csv(&r.columna),
                csv(&r.indicador),
                r.valor,
                csv(&r.lectura)
            )?;
        }
    }

    Ok(())
}

fn escribir_reporte_html(
    path: &Path,
    filas: usize,
    columnas: usize,
    perfil: &[PerfilColumna],
    corrs: &[Correlacion],
    grupos: &[GrupoCorrelacion],
    insights: &[InsightAuto],
    recomendaciones: &[RecomendacionGrafico],
    ml: &[MetricaMl],
    kmeans: Option<&KMeansResultado>,
    montecarlo: Option<&MonteCarloResultado>,
    feature_selection: Option<&FeatureSelectionResultado>,
) -> Result<()> {
    let mut top_nulos: Vec<&PerfilColumna> = perfil.iter().collect();
    top_nulos.sort_by(|a, b| b.porcentaje_nulos.partial_cmp(&a.porcentaje_nulos).unwrap_or(std::cmp::Ordering::Equal));

    let mut top_ml: Vec<&MetricaMl> = ml.iter().collect();
    top_ml.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut html = String::new();
    html.push_str("<!doctype html><html lang='es'><head><meta charset='utf-8'>");
    html.push_str("<title>Reporte analítico Parquet</title>");
    html.push_str(r#"<style>
        body{font-family:Arial,Helvetica,sans-serif;margin:24px;background:#eefbfc;color:#1f2937}
        h1,h2{color:#0f766e}.card{background:#fff8d2;border:1px solid #3c8c96;border-radius:12px;padding:16px;margin:14px 0}
        table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #ddd;padding:8px;font-size:13px}th{background:#32b9c3;color:white;text-align:left}
        .kpi{display:inline-block;background:white;border:1px solid #ddd;border-radius:10px;padding:12px;margin:6px;min-width:150px}
        .bar-bg{height:16px;background:#e5e7eb;border-radius:8px;overflow:hidden}.bar{height:16px;background:#0f766e}
    </style>"#);
    html.push_str("</head><body>");
    html.push_str("<h1>📊 Reporte analítico automático del archivo Parquet</h1>");

    html.push_str("<div class='card'><h2>Resumen</h2>");
    html.push_str(&format!("<div class='kpi'><b>Filas</b><br>{}</div>", filas));
    html.push_str(&format!("<div class='kpi'><b>Columnas</b><br>{}</div>", columnas));
    html.push_str(&format!("<div class='kpi'><b>Correlaciones fuertes</b><br>{}</div>", corrs.len()));
    html.push_str(&format!("<div class='kpi'><b>Grupos</b><br>{}</div>", grupos.len()));
    html.push_str(&format!("<div class='kpi'><b>Insights</b><br>{}</div>", insights.len()));
    let mc_cols = montecarlo.map(|m| m.columnas.len()).unwrap_or(0);
    html.push_str(&format!("<div class='kpi'><b>Monte Carlo</b><br>{} columnas</div>", mc_cols));
    if let Some(fs) = feature_selection {
        html.push_str(&format!("<div class='kpi'><b>Columnas útiles</b><br>{} de {}</div>", fs.columnas_recomendadas, fs.columnas_originales));
    }
    html.push_str("</div>");

    if let Some(fs) = feature_selection {
        html.push_str("<div class='card'><h2>🧹 Feature Selection: selección inteligente de columnas</h2>");
        html.push_str(&format!("<p>{}</p>", html_escape(&fs.conclusion)));
        html.push_str(&format!("<p><b>Aplicado a análisis pesados:</b> {} | <b>Score mínimo:</b> {:.2}</p>", if fs.aplicado_a_analisis { "sí" } else { "no" }, fs.umbral_score));
        html.push_str("<table><tr><th>Decisión</th><th>Score</th><th>Columna</th><th>Tipo</th><th>% nulos</th><th>% únicos</th><th>Corr. máxima</th><th>Motivo</th><th>Acción</th></tr>");
        for x in fs.columnas.iter().take(120) {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:.3}</td><td>{}</td><td>{}</td><td>{:.2}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                html_escape(&x.decision),
                x.score,
                html_escape(&x.columna),
                html_escape(&x.tipo_analitico),
                x.porcentaje_nulos,
                opt_html_f64(x.porcentaje_unicos),
                opt_html_f64(x.correlacion_maxima),
                html_escape(&x.motivo),
                html_escape(&x.accion),
            ));
        }
        html.push_str("</table></div>");
    }

    html.push_str("<div class='card'><h2>Top columnas con más nulos</h2><table><tr><th>Columna</th><th>% nulos</th><th>Visual</th></tr>");
    for p in top_nulos.into_iter().take(20) {
        let w = p.porcentaje_nulos.clamp(0.0, 100.0);
        html.push_str(&format!("<tr><td>{}</td><td>{:.2}</td><td><div class='bar-bg'><div class='bar' style='width:{:.2}%'></div></div></td></tr>", html_escape(&p.columna), p.porcentaje_nulos, w));
    }
    html.push_str("</table></div>");

    html.push_str("<div class='card'><h2>Insights principales</h2><table><tr><th>Severidad</th><th>Categoría</th><th>Columna</th><th>Mensaje</th></tr>");
    for x in insights.iter().take(50) {
        html.push_str(&format!("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>", html_escape(&x.severidad), html_escape(&x.categoria), html_escape(&x.columna), html_escape(&x.mensaje)));
    }
    html.push_str("</table></div>");

    html.push_str("<div class='card'><h2>Correlaciones fuertes</h2><table><tr><th>Columna 1</th><th>Columna 2</th><th>r</th><th>Fuerza</th></tr>");
    for c in corrs.iter().take(100) {
        html.push_str(&format!("<tr><td>{}</td><td>{}</td><td>{:.6}</td><td>{}</td></tr>", html_escape(&c.columna_1), html_escape(&c.columna_2), c.correlacion, html_escape(&c.fuerza)));
    }
    html.push_str("</table></div>");

    if let Some(km) = kmeans {
        html.push_str("<div class='card'><h2>🧩 KMeans: grupos y registros raros</h2>");
        html.push_str(&format!("<p>{}</p>", html_escape(&km.conclusion)));
        html.push_str(&format!("<p><b>Variables usadas:</b> {}</p>", html_escape(&km.features.join(", "))));
        html.push_str("<h3>Clusters</h3><table><tr><th>Cluster</th><th>Cantidad</th><th>%</th><th>Distancia prom.</th><th>Lectura</th></tr>");
        for c in &km.clusters {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{:.2}</td><td>{:.4}</td><td>{}</td></tr>",
                c.cluster,
                c.cantidad,
                c.porcentaje,
                c.distancia_promedio,
                html_escape(&c.lectura)
            ));
        }
        html.push_str("</table>");

        html.push_str("<h3>Registros más raros</h3><table><tr><th>#</th><th>Fila aprox.</th><th>Cluster</th><th>Distancia</th><th>Z</th><th>Severidad</th><th>Explicación</th></tr>");
        for x in km.outliers.iter().take(50) {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.4}</td><td>{:.2}</td><td>{}</td><td>{}</td></tr>",
                x.ranking,
                x.fila_aproximada,
                x.cluster,
                x.distancia,
                x.z_score,
                html_escape(&x.severidad),
                html_escape(&x.valores_principales)
            ));
        }
        html.push_str("</table></div>");
    }

    if let Some(mc) = montecarlo {
        html.push_str("<div class='card'><h2>🎲 Monte Carlo: escenarios posibles</h2>");
        html.push_str(&format!("<p>{}</p>", html_escape(&mc.conclusion)));
        html.push_str(&format!("<p><b>Simulaciones:</b> {} | <b>Tamaño lote:</b> {}</p>", mc.simulaciones, mc.tamano_lote));
        html.push_str("<h3>Resumen por columna</h3><table><tr><th>Columna</th><th>Prom. hist.</th><th>P5 prom.</th><th>P50 prom.</th><th>P95 prom.</th><th>P5 total lote</th><th>P50 total lote</th><th>P95 total lote</th><th>Lectura</th></tr>");
        for c in mc.columnas.iter().take(50) {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.2}</td><td>{:.2}</td><td>{:.2}</td><td>{}</td></tr>",
                html_escape(&c.columna), c.promedio_historico, c.promedio_sim_p5, c.promedio_sim_p50, c.promedio_sim_p95, c.total_sim_p5, c.total_sim_p50, c.total_sim_p95, html_escape(&c.lectura)
            ));
        }
        html.push_str("</table>");
        html.push_str("<h3>Riesgos</h3><table><tr><th>Severidad</th><th>Columna</th><th>Indicador</th><th>Valor</th><th>Lectura</th></tr>");
        for r in mc.riesgos.iter().take(50) {
            html.push_str(&format!("<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.4}</td><td>{}</td></tr>", html_escape(&r.severidad), html_escape(&r.columna), html_escape(&r.indicador), r.valor, html_escape(&r.lectura)));
        }
        html.push_str("</table></div>");
    }

    html.push_str("<div class='card'><h2>Recomendaciones de gráficos</h2><table><tr><th>Prioridad</th><th>Gráfico</th><th>X</th><th>Y</th><th>Razón</th></tr>");
    for x in recomendaciones.iter().take(100) {
        html.push_str(&format!("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>", html_escape(&x.prioridad), html_escape(&x.tipo), html_escape(&x.columna_x), html_escape(&x.columna_y), html_escape(&x.razon)));
    }
    html.push_str("</table></div>");

    html.push_str("<div class='card'><h2>Ranking ML / utilidad analítica</h2><table><tr><th>Columna</th><th>Tipo</th><th>Score</th><th>Recomendación</th><th>Detalle</th></tr>");
    for x in top_ml.into_iter().take(50) {
        html.push_str(&format!("<tr><td>{}</td><td>{}</td><td>{:.3}</td><td>{}</td><td>{}</td></tr>", html_escape(&x.columna), html_escape(&x.tipo), x.score, html_escape(&x.recomendacion), html_escape(&x.detalle)));
    }
    html.push_str("</table></div></body></html>");

    fs::write(path, html)?;
    Ok(())
}


fn opt_html_f64(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.3}"),
        _ => String::new(),
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
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


fn abrir_carpeta_en_sistema(path: &Path) {
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("explorer").arg(path).spawn();
    }

    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(path).spawn();
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = Command::new("xdg-open").arg(path).spawn();
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "perfilador-parquet-entregable-1",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size(egui::vec2(1320.0, 900.0))
                .with_title(format!("📊 Perfilador Parquet Inteligente — {APP_VERSION}")),
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
                acceso_autorizado: false,
                clave_ingresada: String::new(),
                mensaje_acceso: String::new(),
                archivo: None,
                carpeta_salida_base: None,
                umbral_texto: "0.70".into(),
                min_grupo_texto: "3".into(),
                target_texto: String::new(),
                aplicar_feature_selection: true,
                min_score_feature_texto: "0.55".into(),
                proc: Arc::new(Mutex::new(ProcState::default())),
                tab_activa: 0,
            }))
        }),
    )
}
