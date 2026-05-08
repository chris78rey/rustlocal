use std::env;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

fn main() {
    let args: Vec<String> = env::args().collect();
    let dir = if args.len() > 1 {
        args[1].clone()
    } else {
        "/home/crrb/HEALTHC-CONTABILIDAD/Descargas/zzzzzzz".to_string()
    };

    let path = Path::new(&dir);
    if !path.is_dir() {
        eprintln!("Error: '{}' no es un directorio válido", dir);
        std::process::exit(1);
    }

    println!("Escaneando: {}", dir);
    println!("--------------------------------------------------");

    let total_pdfs = Arc::new(AtomicU64::new(0));
    let total_dirs = Arc::new(AtomicU64::new(0));

    // First pass: gather all subdirectories
    let mut subdirs = Vec::new();
    match std::fs::read_dir(path) {
        Ok(entries) => {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    subdirs.push(entry.path());
                }
            }
        }
        Err(e) => {
            eprintln!("Error al leer directorio: {}", e);
            std::process::exit(1);
        }
    }

    subdirs.sort();

    if subdirs.is_empty() {
        println!("(sin subdirectorios, escaneando raíz...)");
        let count = count_pdfs_in_dir(path);
        println!("\nTotal archivos PDF encontrados: {}", count);
        return;
    }

    let total_pdfs_clone = Arc::clone(&total_pdfs);
    let total_dirs_clone = Arc::clone(&total_dirs);
    let dir_count = subdirs.len();

    // Spawn one thread per subdirectory
    let mut handles = Vec::new();
    for subdir in subdirs {
        let total_pdfs = Arc::clone(&total_pdfs_clone);
        let total_dirs = Arc::clone(&total_dirs_clone);
        handles.push(thread::spawn(move || {
            total_dirs.fetch_add(1, Ordering::Relaxed);
            let count = count_pdfs_in_dir(&subdir);
            if count > 0 {
                let name = subdir.file_name().unwrap().to_string_lossy();
                println!("  {:>5} PDFs  |  {}", count, name);
            }
            total_pdfs.fetch_add(count as u64, Ordering::Relaxed);
        }));
    }

    // Wait for all threads
    for h in handles {
        h.join().unwrap();
    }

    let processed = total_dirs_clone.load(Ordering::Relaxed);
    let grand_total = total_pdfs_clone.load(Ordering::Relaxed);

    println!("--------------------------------------------------");
    println!(
        "Directorios procesados: {} / {}",
        processed, dir_count
    );
    println!("Total archivos PDF encontrados: {}", grand_total);
}

fn count_pdfs_in_dir(dir: &Path) -> usize {
    let mut count = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                if p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("pdf"))
                    .unwrap_or(false)
                {
                    count += 1;
                }
            }
        }
    }
    count
}
