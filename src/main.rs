use std::{fs, path::PathBuf};

use ariadne::{Color, Label, Report, ReportKind, Source};
use calamars::{
    parser::parse_module,
    sematic::Resolver,
    source::{SourceDB, SourceFile},
    syntax::token::Token,
};

use chumsky::Parser as chParser; // Chumsky Parser
use clap::{Parser as clParser, Subcommand}; // Clap Parser 

/// Given some file, tokenize, then parser it, and lastly, run the semanics checks.
///
/// Later this will be extended to be the entry point for the calamars compiler and interpreter.
#[derive(Debug, clParser)]
struct CalamarsArgs {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Build a project
    BuildProject {},
    /// Build and run the project
    RunProject {},
    /// Run a single file (interpreter)
    RunFile { path: PathBuf },
}

fn main() {
    match CalamarsArgs::parse().command {
        Commands::BuildProject {} => {
            let db = match SourceDB::load_project() {
                Ok(db) => db,
                Err(e) => {
                    println!("Error: {}", e);
                    return;
                }
            };
            db.analyse_all();
        }
        Commands::RunFile { path } => {
            let file = SourceFile::try_from(path).unwrap();
            let resolver = file.anlayse_file();
            file.display_errors(resolver);
        }
        _ => todo!("Not yet supported"),
    }
}
