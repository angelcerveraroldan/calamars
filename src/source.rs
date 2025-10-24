use std::{
    fs::File,
    io::{BufReader, Read},
};

use chumsky::{Parser, container::Seq, input::Input};

use crate::{
    parser::parse_module,
    sematic::Resolver,
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
            println!("{:?}", err);
        }
    }
}
