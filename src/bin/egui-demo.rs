// ============================================================================
// egui-demo — Ejemplo visual de componentes egui
// Muestra: ventanas, botones, sliders, texto, tabs, tablas, gráficos 2D, etc.
// Ejecutar: cargo run --bin egui-demo
// ============================================================================

use eframe::egui;
use egui::{StrokeKind, Panel};

// ─── Estado de la aplicación ────────────────────────────────────────────────

#[derive(Default)]
struct DemoApp {
    // Botones / check
    counter: i32,
    checked: bool,
    radio_value: String,

    // Sliders
    slider_f32: f32,
    slider_u8: u8,

    // Texto
    name: String,
    multiline: String,

    // Combo
    selected_theme: String,
    themes: Vec<&'static str>,

    // Paint
    current_stroke: Vec<egui::Vec2>,
    strokes: Vec<Vec<egui::Vec2>>,

    // Colapsable / tabs
    tab_index: usize,
    tab_labels: Vec<&'static str>,

    // Grid demo rows
    grid_rows: Vec<[String; 3]>,

    // Progress
    progress: f32,
}

impl DemoApp {
    fn new() -> Self {
        Self {
            counter: 0,
            checked: false,
            radio_value: "Opción A".into(),
            slider_f32: 0.5,
            slider_u8: 128,
            name: "Ecuador".into(),
            multiline: "Escribe\nvarias\nlíneas\nacá...".into(),
            selected_theme: "Claro".into(),
            themes: vec!["Claro", "Oscuro", "Alto Contraste"],
            current_stroke: Vec::new(),
            strokes: Vec::new(),
            tab_index: 0,
            tab_labels: vec!["PDFs", "Texto", "Stats"],
            grid_rows: vec![
                ["PI.pdf".into(), "Planilla Individual".into(), "01-03-2026".into()],
                ["CC.pdf".into(), "Cobertura".into(), "01-03-2026".into()],
                ["008.pdf".into(), "Emergencia".into(), "02-03-2026".into()],
                ["053.pdf".into(), "Referencia".into(), "03-03-2026".into()],
            ],
            progress: 0.33,
        }
    }
}

impl eframe::App for DemoApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // ─── Panel superior (menú / info) ──────────────────────────────────
        let stable_dt = ui.ctx().input(|i| i.stable_dt);
        Panel::top("top_panel").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("🧪 egui Demo — Componentes");
                ui.separator();
                ui.label(format!("dt: {:.1} ms", stable_dt * 1000.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("❌ Salir").clicked() {
                        std::process::exit(0);
                    }
                });
            });
        });

        // ─── Panel izquierdo (navegación) ──────────────────────────────────
        Panel::left("nav_panel")
            .resizable(true)
            .default_size(180.0)
            .show_inside(ui, |ui| {
                ui.heading("📋 Componentes");
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.label("Controles básicos");
                    ui.label("Sliders");
                    ui.label("Texto");
                    ui.label("Desplegables");
                    ui.label("Paint");
                    ui.label("Tabs");
                    ui.label("Tabla");
                    ui.label("Progreso");
                });
            });

        // ─── Panel central ────────────────────────────────────────────────
        let ctx = ui.ctx().clone();
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                // ============================================================
                // SECCIÓN 1: Controles básicos
                // ============================================================
                ui.heading("🖱️  Controles Básicos");
                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button("➕ Incrementar").clicked() {
                        self.counter += 1;
                    }
                    if ui.button("➖ Decrementar").clicked() {
                        self.counter -= 1;
                    }
                    if ui.button("🔄 Resetear").clicked() {
                        self.counter = 0;
                    }
                });
                ui.heading(format!("Valor: **{}**", self.counter));

                ui.add_space(10.0);

                // Checkbox
                ui.checkbox(&mut self.checked, "✅ Casilla de verificación");
                if self.checked {
                    ui.colored_label(egui::Color32::GREEN, "  ✔ Casilla activada");
                }

                ui.add_space(5.0);

                // Radio buttons
                ui.label("🔘 Opción única:");
                ui.radio_value(&mut self.radio_value, "Opción A".to_string(), "Opción A");
                ui.radio_value(&mut self.radio_value, "Opción B".to_string(), "Opción B");
                ui.radio_value(&mut self.radio_value, "Opción C".to_string(), "Opción C");
                ui.label(format!("Seleccionado: {}", self.radio_value));

                ui.add_space(10.0);

                // Botón con tooltip
                let resp = ui.add(
                    egui::Button::new("🛈  Botón con tooltip")
                        .fill(egui::Color32::from_rgb(60, 100, 180))
                );
                if resp.on_hover_text("Este botón tiene un tooltip explicativo.\n¡Los tooltips soportan múltiples líneas!")
                    .clicked()
                {
                    self.counter += 10;
                }

                // ============================================================
                // SECCIÓN 2: Sliders
                // ============================================================
                ui.add_space(20.0);
                ui.heading("🎚️  Sliders");
                ui.separator();

                ui.add(egui::Slider::new(&mut self.slider_f32, 0.0..=1.0)
                    .text("Progreso (f32)")
                    .show_value(true));
                ui.label(format!("  slider_f32 = {:.3}", self.slider_f32));

                ui.add(egui::Slider::new(&mut self.slider_u8, 0..=255)
                    .text("Valor byte (u8)")
                    .show_value(true));
                ui.label(format!("  slider_u8 = {}", self.slider_u8));

                // Progress bar
                ui.add_space(5.0);
                let progress = self.slider_f32;
                let progress_bar = egui::ProgressBar::new(progress)
                    .show_percentage()
                    .animate(true)
                    .fill(if progress < 0.33 {
                        egui::Color32::RED
                    } else if progress < 0.66 {
                        egui::Color32::YELLOW
                    } else {
                        egui::Color32::GREEN
                    });
                ui.add(progress_bar);

                // ============================================================
                // SECCIÓN 3: Texto
                // ============================================================
                ui.add_space(20.0);
                ui.heading("✏️  Entrada de Texto");
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Nombre:");
                    ui.add(egui::TextEdit::singleline(&mut self.name)
                        .hint_text("Escribe algo...")
                        .desired_width(200.0));
                });
                ui.label(format!("Has escrito: **{}**", self.name));

                ui.add_space(5.0);
                ui.label("Texto multilínea:");
                ui.add(egui::TextEdit::multiline(&mut self.multiline)
                    .desired_rows(4)
                    .desired_width(300.0)
                    .lock_focus(true)
                    .font(egui::TextStyle::Monospace));

                // ============================================================
                // SECCIÓN 4: Combo / Dropdown
                // ============================================================
                ui.add_space(20.0);
                ui.heading("📌  Desplegables y Selectores");
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Tema:");
                    egui::ComboBox::from_id_salt("theme_combo")
                        .selected_text(&self.selected_theme)
                        .show_ui(ui, |ui| {
                            for theme in &self.themes {
                                ui.selectable_value(&mut self.selected_theme, (*theme).to_string(), *theme);
                            }
                        });
                });

                // Color picker
                ui.horizontal(|ui| {
                    ui.label("🎨 Color favorito:");
                    let mut color = egui::Color32::BLUE;
                    let _ = egui::color_picker::color_edit_button_srgba(ui, &mut color, egui::color_picker::Alpha::Opaque);
                });

                // ============================================================
                // SECCIÓN 5: Paint (dibujo libre con el mouse)
                // ============================================================
                ui.add_space(20.0);
                ui.heading("🖌️  Dibujo Libre");
                ui.separator();

                ui.label("Dibuja con el mouse en el recuadro:");

                let (paint_resp, painter) = ui.allocate_painter(
                    egui::Vec2::new(400.0, 200.0),
                    egui::Sense::drag(),
                );

                let rect = paint_resp.rect;
                // Fondo
                painter.rect_filled(rect, 4.0, egui::Color32::from_gray(20));
                // Borde
                painter.rect_stroke(rect, 4.0, egui::Stroke::new(1.0, egui::Color32::GRAY), StrokeKind::Outside);

                // Dibujar strokes anteriores
                for stroke in &self.strokes {
                    if stroke.len() >= 2 {
                        let points: Vec<egui::Pos2> = stroke.iter()
                            .map(|v| rect.min + *v)
                            .collect();
                        painter.add(egui::Shape::line(points, egui::Stroke::new(3.0, egui::Color32::LIGHT_BLUE)));
                    }
                }
                // Dibujar stroke actual
                if self.current_stroke.len() >= 2 {
                    let points: Vec<egui::Pos2> = self.current_stroke.iter()
                        .map(|v| rect.min + *v)
                        .collect();
                    painter.add(egui::Shape::line(points, egui::Stroke::new(3.0, egui::Color32::YELLOW)));
                }

                // Capturar mouse
                if let Some(pos) = paint_resp.interact_pointer_pos() {
                    let local = pos - rect.min;
                    if rect.contains(pos) {
                        if paint_resp.dragged() {
                            self.current_stroke.push(local);
                        }
                    }
                }
                if paint_resp.drag_stopped() {
                    if !self.current_stroke.is_empty() {
                        self.strokes.push(std::mem::take(&mut self.current_stroke));
                    }
                }

                ui.horizontal(|ui| {
                    if ui.button("🗑️ Limpiar dibujo").clicked() {
                        self.strokes.clear();
                        self.current_stroke.clear();
                    }
                    ui.label(format!("Trazos: {}", self.strokes.len()));
                });

                // ============================================================
                // SECCIÓN 6: Tabs
                // ============================================================
                ui.add_space(20.0);
                ui.heading("📑  Tabs (Pestañas)");
                ui.separator();

                Panel::bottom("tab_bar")
                    .min_size(0.0)
                    .show_inside(ui, |ui| {
                        ui.horizontal(|ui| {
                            for (i, label) in self.tab_labels.iter().enumerate() {
                                let btn = egui::Button::selectable(self.tab_index == i, *label);
                                if ui.add(btn).clicked() {
                                    self.tab_index = i;
                                }
                            }
                        });
                    });

                // Contenido de cada tab
                egui::Frame::group(ui.style())
                    .inner_margin(egui::Margin::symmetric(8, 8))
                    .show(ui, |ui| {
                        match self.tab_index {
                            0 => {
                                ui.heading("📄 Documentos PDF");
                                ui.label("Lista de PDFs encontrados en el directorio:");
                                for row in &self.grid_rows {
                                    ui.label(format!("  • {} — {} ({})", row[0], row[1], row[2]));
                                }
                            }
                            1 => {
                                ui.heading("🔤 Texto Extraído");
                                ui.label("Aquí iría el texto extraído de los PDFs.");
                                ui.monospace("PLANILLA INDIVIDUAL\nPaciente: GALINDO ALBAN SARAH\nHC: 537088\n...");
                            }
                            2 => {
                                ui.heading("📊 Estadísticas");
                                ui.label("Documentos procesados: 4");
                                ui.label("Páginas totales: 12");
                                ui.label("Tamaño total: 2.4 MB");
                            }
                            _ => {}
                        }
                    });

                // ============================================================
                // SECCIÓN 7: Grid / Table
                // ============================================================
                ui.add_space(20.0);
                ui.heading("📊  Grid / Tabla");
                ui.separator();

                egui::Grid::new("pdf_grid")
                    .striped(true)
                    .min_col_width(100.0)
                    .show(ui, |ui| {
                        // Encabezados
                        ui.strong("Archivo");
                        ui.strong("Descripción");
                        ui.strong("Fecha");
                        ui.end_row();

                        for row in &self.grid_rows {
                            ui.label(&row[0]);
                            ui.label(&row[1]);
                            ui.label(&row[2]);
                            ui.end_row();
                        }
                    });

                // ============================================================
                // SECCIÓN 8: Collapsing Header
                // ============================================================
                ui.add_space(20.0);
                ui.heading("📂  Secciones Plegables");
                ui.separator();

                egui::collapsing_header::CollapsingState::load_with_default_open(
                    ui.ctx(),
                    ui.make_persistent_id("section_1"),
                    true,
                )
                .show_header(ui, |ui| {
                    ui.strong("📁 Detalle del Paciente");
                })
                .body(|ui: &mut egui::Ui| {
                    egui::Grid::new("patient_grid").striped(true).show(ui, |ui| {
                        ui.label("Nombre:"); ui.label("GALINDO ALBAN SARAH ALEJANDRA"); ui.end_row();
                        ui.label("HC:");      ui.label("537088");                      ui.end_row();
                        ui.label("Cédula:");  ui.label("1759260449");                   ui.end_row();
                        ui.label("Edad:");    ui.label("7 años 3 meses");               ui.end_row();
                    });
                });

                egui::CollapsingHeader::new("📈 Datos de Consulta")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.label("Motivo: Infección respiratoria aguda");
                        ui.label("Fecha: 01/03/2026 22:45");
                        ui.label("Médico: VIVAS ARMAS GINA SUSANA");
                    });

                // ============================================================
                // SECCIÓN 9: Ventana flotante
                // ============================================================
                ui.add_space(20.0);
                ui.heading("🪟  Ventana Flotante");
                ui.separator();

                if ui.button("Abrir ventana flotante").clicked() {
                    // La ventana ya está siempre visible con `show(ctx, ...)`
                }

                egui::Window::new("📋 Ventana Flotante")
                    .id(egui::Id::new("floating_window"))
                    .default_pos(egui::pos2(600.0, 300.0))
                    .default_size(egui::vec2(300.0, 200.0))
                    .vscroll(true)
                    .show(&ctx, |ui| {
                        ui.label("Esta es una ventana flotante.");
                        ui.label("Puedes moverla, redimensionarla y cerrarla.");
                        ui.separator();
                        ui.label("Contenido dinámico:");
                        ui.add(egui::Slider::new(&mut self.progress, 0.0..=1.0)
                            .text("Progreso"));
                        let p = egui::ProgressBar::new(self.progress).show_percentage();
                        ui.add(p);
                    });

                // ============================================================
                // SECCIÓN 10: Painting shapes
                // ============================================================
                ui.add_space(20.0);
                ui.heading("🎨  Formas 2D");
                ui.separator();

                let (shape_resp, painter) = ui.allocate_painter(
                    egui::Vec2::new(400.0, 120.0),
                    egui::Sense::hover(),
                );

                let c = shape_resp.rect.center();

                // Rectángulo relleno
                painter.rect_filled(
                    egui::Rect::from_min_size(egui::pos2(c.x - 140.0, c.y - 40.0), egui::vec2(60.0, 60.0)),
                    8.0,
                    egui::Color32::from_rgba_premultiplied(100, 150, 200, 180),
                );

                // Círculo
                painter.circle_filled(
                    egui::pos2(c.x - 40.0, c.y),
                    30.0,
                    egui::Color32::from_rgb(200, 100, 100),
                );

                // Triángulo (polígono convexo)
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        egui::pos2(c.x + 40.0, c.y - 35.0),
                        egui::pos2(c.x + 10.0, c.y + 35.0),
                        egui::pos2(c.x + 80.0, c.y + 20.0),
                    ],
                    egui::Color32::from_rgb(80, 200, 80),
                    egui::Stroke::new(2.0, egui::Color32::WHITE),
                ));

                // Estrella (texto con fondo)
                painter.text(
                    egui::pos2(c.x + 160.0, c.y),
                    egui::Align2::CENTER_CENTER,
                    "⭐",
                    egui::FontId::proportional(40.0),
                    egui::Color32::WHITE,
                );

                // Etiquetas
                painter.text(
                    egui::pos2(c.x - 110.0, c.y + 50.0),
                    egui::Align2::CENTER_CENTER,
                    "Rect",
                    egui::FontId::proportional(12.0),
                    egui::Color32::LIGHT_GRAY,
                );
                painter.text(
                    egui::pos2(c.x - 40.0, c.y + 50.0),
                    egui::Align2::CENTER_CENTER,
                    "Circle",
                    egui::FontId::proportional(12.0),
                    egui::Color32::LIGHT_GRAY,
                );
                painter.text(
                    egui::pos2(c.x + 50.0, c.y + 50.0),
                    egui::Align2::CENTER_CENTER,
                    "Poly",
                    egui::FontId::proportional(12.0),
                    egui::Color32::LIGHT_GRAY,
                );
            });
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::vec2(900.0, 700.0))
            .with_title("🧪 egui Demo — Componentes GUI en Rust"),
        ..Default::default()
    };

    eframe::run_native(
        "egui-demo",
        options,
        Box::new(|_cc| Ok(Box::new(DemoApp::new()))),
    )
}
