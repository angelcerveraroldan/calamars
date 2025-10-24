use std::{fs, path::PathBuf};

use ariadne::{Color, Label, Report, ReportKind, Source};
use calamars::{parser::parse_module, sematic::Resolver, source::SourceFile, syntax::token::Token};

use chumsky::Parser as chParser; // Chumsky Parser
use clap::Parser as clParser; // Clap Parser 

/// Given some file, tokenize, then parser it, and lastly, run the semanics checks.
///
/// Later this will be extended to be the entry point for the calamars compiler and interpreter.
#[derive(Debug, clParser)]
struct CalamarsArgs {
    /// Path to the source file
    source_file: PathBuf,
}

fn main() {
    let args = CalamarsArgs::parse();
    let file = SourceFile::try_from(args.source_file).unwrap();
    let resolver = file.anlayse_file();
    file.display_errors(resolver);
}
