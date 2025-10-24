use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
};

use ariadne::{Label, Report, ReportKind, Source};
use chumsky::{Parser, container::Seq, input::Input};
use proptest::collection::HashMapStrategy;

use crate::{
    parser::parse_module,
    sematic::{Resolver, error::SemanticError},
    syntax::{
        ast::{self, TokenInput},
        token::Token,
    },
};

pub struct FileId(pub usize);

/// A Calamars source file
pub struct SourceFile {
    pub id: FileId,
    pub path: std::path::PathBuf,
    pub src: String,
}

impl TryFrom<std::path::PathBuf> for SourceFile {
    type Error = std::io::Error;

    fn try_from(path: std::path::PathBuf) -> Result<Self, Self::Error> {
        let file = File::open(&path)?;
        let mut buf_reader = BufReader::new(file);
        let mut src = String::new();
        buf_reader.read_to_string(&mut src)?;
        let id = FileId(0);
        Ok(SourceFile { id, path, src })
    }
}

impl SourceFile {
    fn file_name(&self) -> String {
        self.path
            .file_name()
            .map(|os_str| os_str.to_str())
            .flatten()
            .unwrap()
            .to_string()
    }

    fn file_source(&self) -> &String {
        &self.src
    }

    /// Tokenize the file, and return the tokens as a TokenInput stream
    pub fn as_spanned_token_stream(&self) -> impl TokenInput<'_> {
        Token::tokens_spanned_stream(&self.src)
    }

    pub fn parse_file(&self) -> (Option<ast::Module>, Vec<chumsky::prelude::Rich<'_, Token>>) {
        let tks = self.as_spanned_token_stream();
        parse_module().parse(tks).into_output_errors()
    }

    pub fn anlayse_file(&self) -> Resolver {
        let (outs, errs) = self.parse_file();
        for err in errs {
            println!("{:?}", err);
        }
        let mut resolver = Resolver::default();
        resolver.verify_module(&outs.expect("Make sure parsing works..."));
        resolver
    }

    pub fn display_errors(&self, resolver: Resolver) {
        for err in resolver.errors() {
            self.log_error(err.clone());
        }
    }

    /// Given a semantic error, pretty print it with the source code
    pub fn log_error(&self, error: SemanticError) -> Result<(), std::io::Error> {
        let file_name = self.file_name();
        let rep = Report::build(ReportKind::Error, (&file_name, 12..12))
            .with_message(error.main_message().to_string());

        let mut rep_labelled = error
            .ariadne_labels(&file_name)
            .into_iter()
            .fold(rep, |rep, label| rep.with_label(label));

        if let Some(note) = error.notes() {
            rep_labelled = rep_labelled.with_note(note);
        };

        let fin = rep_labelled.finish();
        fin.print((&file_name, Source::from(self.file_source())))
    }
}
