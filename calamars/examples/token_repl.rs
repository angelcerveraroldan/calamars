use std::io::{Write, stdin, stdout};

use ariadne::{Color, Label, Report, ReportKind, Source};
use calamars::{parser::base_type_parser, token::Token};
use chumsky::prelude::*;
use chumsky::{input::Stream, prelude::*, span::SimpleSpan};
use logos::Logos;
use std::{collections::HashMap, env, fs};

fn main() {
    println!("Type a Calamars line. I’ll show tokens and a parsed AST.\n");
    loop {
        print!(">>> ");
        let _ = stdout().flush();

        let mut line = String::new();
        if stdin().read_line(&mut line).is_err() {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }

        let tokens = Token::tokenize_line(&line);
        println!("{:?}", tokens);

        let stream = Token::tokens_spanned_stream(&line);
        let (out, errs) = base_type_parser().parse(stream).into_output_errors();
        println!("{:?}", out);
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
                .print(Source::from(&line))
                .unwrap()
        });
    }
}
