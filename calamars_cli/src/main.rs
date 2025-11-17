use calamars_core::ids;
use clap::{Parser, Subcommand};
use front::{
    errors::PrettyError,
    sematic::{lower::HirBuilder, types::TypeHandler},
    syntax::parser::CalamarsParser,
};
use ir::printer::MirPrinter;
use std::path::PathBuf;

use crate::source::SourceFile;

mod source;

#[derive(Parser, Debug)]
#[command(
    name = "calamars",
    version,
    about = "Calamars language tools",
    author = "Angel Cervera Roldan"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Build a Calamars project / file
    Build {
        /// Emit MIR instead of (or before) other outputs
        #[arg(long)]
        mir: bool,

        /// Path to the source file or project root
        #[arg(value_name = "SOURCE_PATH")]
        path: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Build { mir, path } => {
            let sf = SourceFile::try_from((0, path)).expect("Failed to read source file");
            let tokens = sf.as_spanned_token_stream();

            let file_id = ids::FileId::from(0);
            let file_name = String::from("tmp");

            let mut parser = CalamarsParser::new(file_id, tokens);
            let module: front::syntax::ast::Module = parser.parse_file();

            let mut module = match HirBuilder::lower_module(&module, file_id, file_name.clone()) {
                Ok(module) => module,
                Err(errors) => {
                    for err in errors {
                        err.log_error(&file_name, &sf.src);
                    }
                    return;
                }
            };

            // Type checking
            let mut type_handler = TypeHandler {
                module: &mut module,
                errors: vec![],
            };
            type_handler.type_check_module();

            if !type_handler.errors.is_empty() {
                for err in type_handler.errors {
                    err.log_error(&file_name, &sf.src);
                }
                return;
            }

            // Lower to HIR
            let mut mir_builder = ir::lower::MirBuilder::new(&module);
            match mir_builder.lower_module() {
                Ok(_) => {}
                Err(errs) => {
                    for err in errs {
                        println!("{err:?}");
                    }
                }
            }

            // Print all the functions!
            let printer = MirPrinter::new(
                mir_builder.blocks(),
                mir_builder.instructions(),
                mir_builder.functions(),
            );
            let s = printer.fmt_all_functions();
            println!("{s}");
        }
    }
}
