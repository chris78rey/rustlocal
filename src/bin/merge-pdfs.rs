use lopdf::{Document, Object};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

fn merge_pdfs(inputs: &[String]) -> Result<Document, Box<dyn std::error::Error>> {
    let version = "1.7";
    let mut merged = Document::with_version(version);
    let mut max_id: u32 = 1;

    // All Page objects collected separately
    let mut all_pages: BTreeMap<lopdf::ObjectId, lopdf::Object> = BTreeMap::new();

    // The first Catalog and Pages root we encounter
    let mut catalog_obj: Option<(lopdf::ObjectId, lopdf::Object)> = None;
    let mut pages_root_obj: Option<(lopdf::ObjectId, lopdf::Object)> = None;

    for pdf_path in inputs {
        let mut doc = Document::load(pdf_path)?;

        // Renumber objects to avoid ID collisions
        doc.renumber_objects_with(max_id);
        max_id = doc.max_id + 1;

        // Separate Page objects from everything else
        let pages_in_doc = doc.get_pages();

        for (_, page_id) in &pages_in_doc {
            if let Ok(obj) = doc.get_object(*page_id) {
                all_pages.insert(*page_id, obj.clone());
            }
        }

        for (id, obj) in std::mem::take(&mut doc.objects) {
            // Skip Page objects (handled separately)
            if pages_in_doc.values().any(|pid| *pid == id) {
                continue;
            }

            match obj.type_name().unwrap_or("") {
                "Catalog" => {
                    catalog_obj.get_or_insert((id, obj));
                }
                "Pages" => {
                    if let Ok(dict) = obj.as_dict() {
                        let mut d = dict.clone();
                        if let Some((_, ref existing)) = pages_root_obj {
                            if let Ok(old_dict) = existing.as_dict() {
                                // Merge Kids lists
                                if let Ok(Ok(old_kids)) = old_dict.get(b"Kids").map(|k| k.as_array()) {
                                    if let Ok(Ok(new_kids)) = d.get(b"Kids").map(|k| k.as_array()) {
                                        let mut combined = old_kids.clone();
                                        combined.extend(new_kids.iter().cloned());
                                        d.set("Kids", combined);
                                    }
                                }
                            }
                        }
                        pages_root_obj.get_or_insert((id, Object::Dictionary(d.clone())));
                        // Keep the latest dict (with merged Kids)
                        pages_root_obj = Some((pages_root_obj.unwrap().0, Object::Dictionary(d)));
                    }
                }
                _ => {
                    merged.objects.insert(id, obj);
                }
            }
        }
    }

    // Ensure we have a Pages root
    let (pages_root_id, pages_root_obj) = pages_root_obj
        .ok_or_else(|| "No se encontró Pages root en ningún PDF".to_string())?;

    // Build final Kids list from collected pages and set Count
    let kids: Vec<lopdf::Object> = all_pages
        .keys()
        .map(|id| lopdf::Object::Reference(*id))
        .collect();

    let page_count = kids.len() as u32;

    if let Ok(dict) = pages_root_obj.as_dict() {
        let mut d = dict.clone();
        d.set("Kids", kids);
        d.set("Count", page_count);
        merged.objects.insert(pages_root_id, lopdf::Object::Dictionary(d));
    }

    // Insert all Page objects with updated Parent reference
    for (id, obj) in all_pages {
        if let Ok(dict) = obj.as_dict() {
            let mut d = dict.clone();
            d.set("Parent", pages_root_id);
            merged.objects.insert(id, lopdf::Object::Dictionary(d));
        }
    }

    // Set Catalog to point to Pages root
    if let Some((catalog_id, catalog_obj)) = catalog_obj {
        if let Ok(dict) = catalog_obj.as_dict() {
            let mut d = dict.clone();
            d.set("Pages", pages_root_id);
            d.remove(b"Outlines");
            merged.objects.insert(catalog_id, lopdf::Object::Dictionary(d));
        }
        merged.trailer.set("Root", catalog_id);
    }

    merged.max_id = merged.objects.len() as u32;
    merged.renumber_objects();

    Ok(merged)
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let dir = if args.len() > 1 {
        args[1].clone()
    } else {
        "/home/crrb/HEALTHC-CONTABILIDAD/Descargas/zzzzzzz/5867345".to_string()
    };

    let dir_path = Path::new(&dir);
    if !dir_path.is_dir() {
        eprintln!("Error: '{}' no es un directorio válido", dir);
        std::process::exit(1);
    }

    // Collect all .pdf files sorted
    let mut pdfs: Vec<String> = Vec::new();
    match fs::read_dir(dir_path) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext.eq_ignore_ascii_case("pdf") {
                            let fname = path.file_name().unwrap().to_string_lossy().to_string();
                            if fname.eq_ignore_ascii_case("todo.pdf") {
                                continue;
                            }
                            pdfs.push(path.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Error al leer el directorio: {}", e);
            std::process::exit(1);
        }
    }

    pdfs.sort();

    if pdfs.is_empty() {
        eprintln!("No se encontraron archivos PDF en '{}'", dir);
        std::process::exit(1);
    }

    println!("PDFs encontrados ({}):", pdfs.len());
    for (i, p) in pdfs.iter().enumerate() {
        let name = Path::new(p).file_name().unwrap().to_string_lossy();
        println!("  {}. {}", i + 1, name);
    }

    println!("\nFusionando PDFs (100% Rust con lopdf)...");

    match merge_pdfs(&pdfs) {
        Ok(mut merged_doc) => {
            let output = dir_path.join("todo.pdf");
            let output_str = output.to_string_lossy().to_string();

            match merged_doc.save(&output_str) {
                Ok(_) => {
                    println!("✅ PDF fusionado creado: {}", output_str);
                    if let Ok(meta) = fs::metadata(&output_str) {
                        println!("   Tamaño: {} bytes ({} KB)", meta.len(), meta.len() / 1024);
                    }
                    println!("\nTamaños originales:");
                    let mut total_original: u64 = 0;
                    for p in &pdfs {
                        if let Ok(m) = fs::metadata(p) {
                            let name = Path::new(p).file_name().unwrap().to_string_lossy();
                            println!("   {:>8} KB  {}", m.len() / 1024, name);
                            total_original += m.len();
                        }
                    }
                    println!("   ─────────────");
                    println!("   {:>8} KB  total original", total_original / 1024);
                }
                Err(e) => {
                    eprintln!("Error al guardar el PDF: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("Error durante la fusión: {}", e);
            std::process::exit(1);
        }
    }
}
