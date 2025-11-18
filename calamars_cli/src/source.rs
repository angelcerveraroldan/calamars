use std::{
    collections::BTreeSet,
    env,
    fmt::Debug,
    fs::{self, File},
    io::{BufReader, Read},
    path::PathBuf,
};

use calamars_core::ids;
use front::syntax;

#[derive(Debug, Clone, Copy)]
pub struct FileId(pub usize);

/// A Calamars source file
#[derive(Debug)]
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
    pub fn repl(source: String) -> Self {
        Self {
            id: FileId(0),
            path: PathBuf::from("repl.cm"),
            src: source,
        }
    }

    fn file_name(&self) -> String {
        self.path
            .file_name()
            .and_then(|os_str| os_str.to_str())
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
    pub fn as_spanned_token_stream(&self) -> Vec<(syntax::token::Token, logos::Span)> {
        syntax::token::Token::tokens_spanned_stream(&self.src)
    }

    pub fn parse_file(&self) -> (syntax::ast::Module, syntax::parser::CalamarsParser) {
        let tks = self.as_spanned_token_stream();
        let mut parser = syntax::parser::CalamarsParser::new(ids::FileId::from(0), tks);
        (parser.parse_file(), parser)
    }
}

/// All of the projects files
///
/// Once this is created with `SourceDBBuilder`, it cannot be mutated (as it can break ordering of
/// vector)
pub struct SourceDB {
    files: Vec<SourceFile>,
    config_file: SourceFile,
}

impl SourceDB {
    pub fn load_project() -> Result<Self, std::io::Error> {
        let root = env::current_dir()?;
        let config_path = root.join("project.cm");
        if !config_path.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No `project.cm` in the current directory. Run from the project root.",
            ));
        }

        // Load config file
        let mut builder = SourceDBBuilder::default().add_config(config_path);

        // Walk ./src recursively, collecting *.cm files
        let src_dir = root.join("src");
        if !src_dir.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Expected `src/` directory in the project root.",
            ));
        }

        // Simple manual DFS to avoid extra deps
        let mut stack = vec![src_dir];
        while let Some(dir) = stack.pop() {
            for entry in fs::read_dir(&dir)? {
                let entry = entry?;
                let path = entry.path();
                let ftype = entry.file_type()?;

                if ftype.is_dir() {
                    // Skip hidden directories like .git
                    if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                        if name.starts_with('.') {
                            continue;
                        }
                    }
                    stack.push(path);
                } else if ftype.is_file() {
                    // Only *.cm files; skip any stray project.cm in src
                    let is_cm = path.extension().and_then(|s| s.to_str()) == Some("cm");
                    let is_project =
                        path.file_name().and_then(|s| s.to_str()) == Some("project.cm");
                    if is_cm && !is_project {
                        let canon = path.canonicalize()?;
                        builder = builder.add_path(canon);
                    }
                }
            }
        }

        builder.finish()
    }
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

        let config = TryFrom::try_from(builder.config.expect("Config needed"))?;

        Ok(Self {
            files,
            config_file: config,
        })
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
    config: Option<PathBuf>,
}

impl SourceDBBuilder {
    pub fn add_path(mut self, new_path: PathBuf) -> Self {
        self.file_paths.insert(new_path);
        self
    }

    pub fn add_config(mut self, config: PathBuf) -> Self {
        self.config = Some(config);
        self
    }

    pub fn finish(self) -> Result<SourceDB, std::io::Error> {
        TryFrom::try_from(self)
    }
}
