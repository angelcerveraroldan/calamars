use std::path::PathBuf;

use ariadne::Source;
use clap::builder::PathBufValueParser;
use front::{sematic::lower::HirBuilder, source::SourceFile, syntax::token::Token};

#[test]
fn load_minimal() {
    const MINIMAL_FILE: &'static str = "../docs/minimal.cm";
    let path = PathBuf::from(MINIMAL_FILE);
    let source_file = SourceFile::try_from(path).expect("Could not open file");
    let (module, parser) = source_file.parse_file();

    // Analysie the file now
    let mut hb = HirBuilder::default();
    // Load in the module
    hb.module(&module);

    for e in hb.errors() {
        println!("{:?}", e);
    }

    // Looking at minimial.cm, after lowering to HIR, but not type checking yet, we should expect
    // two errors (1, 5), both of which are re-declaration errors.
    let expected = 2;
    assert_eq!(hb.errors().len(), expected)
}
