use std::{
    collections::{BTreeSet, HashMap},
    fmt::Debug,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use ariadne::{Color, Label, Report, ReportKind, Source};
use chumsky::{Parser, container::Seq, input::Input};
use clap::builder::PathBufValueParser;
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
    pub path: PathBuf,
    pub src: String,
}

impl TryFrom<PathBuf> for SourceFile {
    type Error = std::io::Error;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        let file = File::open(&path)?;
        let mut buf_reader = BufReader::new(file);
        let mut src = String::new();
        buf_reader.read_to_string(&mut src)?;
        let id = FileId(0);
        Ok(SourceFile { id, path, src })
    }
}

impl TryFrom<(usize, PathBuf)> for SourceFile {
    type Error = std::io::Error;

    fn try_from((id, path): (usize, PathBuf)) -> Result<Self, Self::Error> {
        let file = File::open(&path)?;
        let mut buf_reader = BufReader::new(file);
        let mut src = String::new();
        buf_reader.read_to_string(&mut src)?;
        let id = FileId(id);
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

    fn path(&self) -> &PathBuf {
        &self.path
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
                .print(Source::from(&self.src))
                .unwrap()
        });

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

/// All of the projects files
///
/// Once this is created with `SourceDBBuilder`, it cannot be mutated (as it can break ordering of
/// vector)
pub struct SourceDB {
    files: Vec<SourceFile>,
}

impl TryFrom<SourceDBBuilder> for SourceDB {
    type Error = std::io::Error;

    fn try_from(builder: SourceDBBuilder) -> Result<Self, Self::Error> {
        let files = builder
            .file_paths
            .into_iter()
            .enumerate()
            .map(TryFrom::try_from)
            .collect::<Result<Vec<SourceFile>, Self::Error>>()?;
        Ok(Self { files })
    }
}

impl SourceDB {
    pub fn files(&self) -> &Vec<SourceFile> {
        &self.files
    }

    pub fn get_file(&self, FileId(id): FileId) -> Option<&SourceFile> {
        self.files.get(id)
    }

    pub fn get_path(&self, id: FileId) -> Option<&PathBuf> {
        self.get_file(id).map(|file| &file.path)
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }
}

#[derive(Default, Debug)]
pub struct SourceDBBuilder {
    file_paths: BTreeSet<PathBuf>,
}

impl SourceDBBuilder {
    pub fn add_path(mut self, new_path: PathBuf) -> Self {
        self.file_paths.insert(new_path);
        self
    }

    pub fn finish(self) -> Result<SourceDB, std::io::Error> {
        TryFrom::try_from(self)
    }
}

#[cfg(test)]
mod test_builder {
    use std::{fs, path::PathBuf};

    use crate::source::{FileId, SourceDBBuilder};

    #[test]
    fn load_many_files() {
        let sourcedb = SourceDBBuilder::default()
            .add_path("./tests/test_files/sample_file.cm".into())
            .add_path("./tests/test_files/bad_function.cm".into())
            .finish();

        assert!(sourcedb.is_ok());
        let sourcedb = sourcedb.unwrap();
        assert_eq!(sourcedb.len(), 2);
        assert_eq!(
            sourcedb.get_file(FileId(0)).unwrap().path().clone(),
            PathBuf::from("./tests/test_files/bad_function.cm"),
            "Paths shuold be inserted in alphabetical order"
        );
    }
}
