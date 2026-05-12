// Pequeño helper para inspeccionar un Parquet
use polars::prelude::*;
use std::env;
use std::fs::File;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = if args.len() > 1 { &args[1] } else { "/home/crrb/datos_digitalizacion.parquet" };

    let mut file = File::open(path).unwrap();
    let df = ParquetReader::new(&mut file).finish().unwrap();

    println!("Archivo: {path}");
    println!("Filas: {}, Columnas: {}", df.height(), df.width());
    println!();

    for col in df.columns() {
        let s: &Series = col.as_materialized_series();
        println!("  {:30} {:20} nulos={}",
            s.name(), format!("{:?}", s.dtype()), s.null_count());
    }
}
