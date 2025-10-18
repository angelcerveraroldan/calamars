use std::fs;

use ariadne::{Color, Label, Report, ReportKind, Source};
use calamars::{parser::parse_module, sematic::Resolver, syntax::token::Token};

use chumsky::Parser as chParser; // Chumsky Parser
use clap::Parser as clParser; // Clap Parser 

/// Given some file, tokenize, then parser it, and lastly, run the semanics checks.
///
/// Later this will be extended to be the entry point for the calamars compiler and interpreter.
#[derive(Debug, clParser)]
struct CalamarsArgs {
    /// Path to the source file
    source_file: String,
}

fn main() {
    let args = CalamarsArgs::parse();
    let src = fs::read_to_string(args.source_file).expect("Did not find file");

    // First tokenize the file, and turn it into a stream
    let token_stream = Token::tokens_spanned_stream(&src);
    let (out, errs) = parse_module().parse(token_stream).into_output_errors();
    errs.into_iter().for_each(|e| {
        Report::build(ReportKind::Error, ((), e.span().into_range()))
            .with_config(ariadne::Config::new().with_index_type(ariadne::IndexType::Byte))
            .with_message(e.to_string())
            .with_label(
                Label::new(((), e.span().into_range()))
                    .with_message(e.reason().to_string())
                    .with_color(Color::Red),
            )
            .finish()
            .print(Source::from(&src))
            .unwrap()
    });

    // We didnt get a module, so we cannot go onto semantic analysis
    if out.is_none() {
        return;
    }

    let out = out.unwrap();
    let mut module_resolver = Resolver::default();

    for item in out.items {
        match item {
            calamars::syntax::ast::ClItem::Declaration(cl_declaration) => {
                let r = module_resolver.push_ast_declaration(&cl_declaration);
            }
            calamars::syntax::ast::ClItem::Expression(cl_expression) => {
                println!("Expressions not yet handled")
            }
            calamars::syntax::ast::ClItem::Import => println!("Imports are not yet supported"),
        }
    }

    // Now we display the errors:
    for error in module_resolver.errors() {
        println!("{:?}", error);
    }
}
