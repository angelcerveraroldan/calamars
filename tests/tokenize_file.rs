use calamars::token::Token;
use logos::Logos;
use std::fs;

#[test]
fn tokenize_file() {
    let source = fs::read_to_string("tests/test_files/sample_file.cal").unwrap();
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
            Equal,
            Ident("x".to_string()),
            Plus,
            Ident("y".to_string()),
            Semicolon,
            // var list : List[int] = [1, 2, 3, 4, 5];
            LineComment,
            Var,
            Ident("list".to_string()),
            Colon,
            Ident("List".to_string()),
            LBracket,
            Ident("int".to_string()),
            RBracket,
            Equal,
            LBracket,
            Int(1),
            Comma,
            Int(2),
            Comma,
            Int(3),
            Comma,
            Int(4),
            Comma,
            Int(5),
            RBracket,
            Semicolon,
        ]
    };

    assert_eq!(actual, expected)
}
