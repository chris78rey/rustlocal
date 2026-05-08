use std::env;
use std::path::Path;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();

    let pdf_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "/home/crrb/HEALTHC-CONTABILIDAD/Descargas/zzzzzzz/5867345/08.pdf".to_string()
    };

    let txt_output = if args.len() > 2 {
        args[2].clone()
    } else {
        let p = Path::new(&pdf_path);
        let stem = p.file_stem().unwrap_or_default().to_string_lossy();
        format!("{}.txt", stem)
    };

    if !Path::new(&pdf_path).exists() {
        eprintln!("Error: no se encuentra el PDF '{}'", pdf_path);
        std::process::exit(1);
    }

    println!("Extrayendo texto de: {}", pdf_path);

    let bytes = match fs::read(&pdf_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error al leer el PDF: {}", e);
            std::process::exit(1);
        }
    };

    let text = match pdf_extract::extract_text_from_mem(&bytes) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error al extraer texto del PDF: {}", e);
            std::process::exit(1);
        }
    };

    match fs::write(&txt_output, &text) {
        Ok(_) => {
            println!("✅ Texto extraído exitosamente a: {}", txt_output);
            let lines = text.lines().count();
            let chars = text.chars().count();
            println!("   {} líneas, {} caracteres", lines, chars);
        }
        Err(e) => {
            eprintln!("Error al escribir el archivo TXT: {}", e);
            std::process::exit(1);
        }
    }
}
