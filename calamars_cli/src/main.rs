use calamars_core::ids;
use clap::{Parser, Subcommand};
use front::{
    errors::PrettyError,
    sematic::{lower::HirBuilder, types::TypeHandler},
    syntax::parser::CalamarsParser,
};
use ir::printer::MirPrinter;
use std::path::PathBuf;

use calamars_cli::source::SourceFile;

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
        /// Emit MIR
        #[arg(long, alias = "mir")]
        emit_mir: bool,

        /// Run the program on the VM
        #[arg(long)]
        run_vm: bool,

        /// Path to the source file or project root
        #[arg(value_name = "SOURCE_PATH")]
        path: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Build {
            emit_mir,
            run_vm,
            path,
        } => {
            let sf = SourceFile::try_from((0, path)).expect("Failed to read source file");
            let tokens = sf.as_spanned_token_stream();

            let file_id = ids::FileId::from(0);
            let file_name = String::from("tmp");

            let mut parser = CalamarsParser::new(file_id, tokens);
            let module: front::syntax::ast::Module = parser.parse_file();

            if !parser.diag().is_empty() {
                for err in parser.diag() {
                    err.log_error(&file_name, &sf.src);
                }
                std::process::exit(1);
            }

            let (mut module, errors) =
                HirBuilder::lower_module(&module, file_id, file_name.clone());

            if !errors.is_empty() {
                for err in errors {
                    err.log_error(&file_name, &sf.src);
                }
                std::process::exit(1);
            }

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
                std::process::exit(1);
            }

            // Lower to MIR
            let mut mir_builder = ir::lower::FunctionBuilder::new(&module);
            let mut funcs = Vec::new();
            for symbol_id in &module.roots {
                let symbol = module.symbols.get_unchecked(*symbol_id);
                let name = symbol.ident_id();
                let return_ty = symbol.ty_id();

                let (params, body) = if let front::sematic::hir::SymbolKind::Function {
                    params,
                    body,
                } = &symbol.kind
                {
                    (params, body)
                } else {
                    continue;
                };

                match mir_builder.lower(name, return_ty, params, *body) {
                    Ok(fun) => funcs.push(fun),
                    Err(_) => continue,
                }
            }

            if emit_mir {
                let printer = MirPrinter::new(&funcs);
                let s = printer.fmt_all_functions();
                println!("{s}");
            }

            if run_vm {
                let irmodule = ir::Module { functions: funcs };
                let mut vmlower = vm::Lowerer::new(&irmodule);
                println!("{:?}", vmlower.run_module());
            }
        }
    }
}
