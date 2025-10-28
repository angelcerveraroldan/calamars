use std::{
    env,
    fs::{self, File},
    io::Write,
    path::PathBuf,
};

use ariadne::{Color, Label, Report, ReportKind, Source};
use calamars::{
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
    /// Create a new project
    NewProject {
        /// Name of the new project
        name: String,
    },
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
        Commands::NewProject { name } => {
            let defmain = "def main() = {}";
            let defproj = "-- This is where your config goes!";

            let current_path = env::current_dir().unwrap();
            let project_path = current_path.join(&name);
            fs::create_dir(&project_path);

            let src_path = project_path.join("src");
            fs::create_dir(&src_path);

            let mut proj = File::create_new(project_path.join("project.cm")).unwrap();
            let mut main = File::create_new(src_path.join("main.cm")).unwrap();
            proj.write_all(defproj.as_bytes());
            main.write_all(defmain.as_bytes());
        }
        _ => todo!("Not yet supported"),
    }
}
