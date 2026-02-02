use calamars_core::ids;
use clap::{Parser, Subcommand};
use front::{
    errors::PrettyError,
    sematic::{lower::HirBuilder, types::TypeHandler},
    syntax::parser::CalamarsParser,
};
use ir::printer::MirPrinter;
use std::path::PathBuf;
use vm::VMachine;

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

            let mut mir_builder = ir::lower::ModuleBuilder::new(&module);
            mir_builder.lower_entire_module().expect("lowering failed");
            let irmodule = mir_builder.finish();

            if emit_mir {
                let printer = MirPrinter::new(irmodule.function_arena.inner());
                println!("{}", printer.fmt_all_functions());
            }

            if run_vm {
                let mut vmlower = vm::lower::Lowerer::new(&irmodule);
                let functions = vmlower
                    .lower_module()
                    .map_err(|err| {
                        format!(
                            "Failed to lower from MIR to VM Bytecode with error: {:?}",
                            err
                        )
                    })
                    .unwrap();

                let mut vm = VMachine::new(functions.into_boxed_slice(), ir::FunctionId::from(0))
                    .expect("Failed to lower to vm");
                let out = vm.run();
                println!("Main fn returns: {:?}", out);
            }
        }
    }
}
