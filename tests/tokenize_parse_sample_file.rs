use calamars::{
    parser::parse_module,
    syntax::{
        ast::{ClDeclaration, ClFuncDec},
        token::Token,
    },
};
use chumsky::Parser;
use logos::Logos;
use std::fs;

#[test]
fn tokenize_file() {
    let source = fs::read_to_string("tests/test_files/sample_file.cm").unwrap();
    let mut lex = Token::lexer(source.as_str());
    let mut actual = Vec::new();
    while let Some(tok) = lex.next() {
        match tok {
            Ok(t) => actual.push(t),
            Err(_) => panic!("lexing error at byte span {:?}", lex.span()),
        }
    }

    let expected = {
        use Token::*;
        vec![
            // val x : int = 2;
            Val,
            Ident("x".to_string()),
            Colon,
            Ident("int".to_string()),
            Equal,
            Int(2),
            Semicolon,
            // var y : int = 3;
            Var,
            Ident("y".to_string()),
            Colon,
            Ident("int".to_string()),
            Equal,
            Int(3),
            Semicolon,
            // def add(x: int, y: int) = x + y;
            DocComment(" Add two integers! ".to_string()),
            Def,
            Ident("add".to_string()),
            LParen,
            Ident("x".to_string()),
            Colon,
            Ident("int".to_string()),
            Comma,
            Ident("y".to_string()),
            Colon,
            Ident("int".to_string()),
            RParen,
            Colon,
            Ident("int".to_string()),
            Equal,
            Ident("x".to_string()),
            Plus,
            Ident("y".to_string()),
            // var list : List[int] = [1, 2, 3, 4, 5];
            Var,
            Ident("list".to_string()),
            Colon,
            Ident("string".to_string()),
            Equal,
            String("hello".to_string()),
            Semicolon,
        ]
    };

    assert_eq!(actual, expected)
}

#[test]
fn parse_file() {
    let source = fs::read_to_string("tests/test_files/sample_file.cm").unwrap();
    let stream = Token::tokens_spanned_stream(&source);
    let (out, errs) = parse_module().parse(stream).into_output_errors();

    for err in errs {
        println!("{:?}", err);
    }

    // Parsing should pass
    assert!(out.is_some());

    let out = out.unwrap();
    assert!(out.items.len() == 4);

    let finalf = match out.items[2].get_dec() {
        ClDeclaration::Binding(cl_binding) => todo!(),
        ClDeclaration::Function(cl_func_dec) => cl_func_dec,
    };

    /// Check that the parser is handling the doc comment properly
    assert_eq!(finalf.doc_comment, Some(" Add two integers! ".to_string()));
}
