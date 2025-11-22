use calamars_core::ids::FileId;
use front::syntax::{
    ast::{Declaration, Expression, Item},
    token::Token,
};

#[test]
fn parse_fn_without_spaces() {
    let source = "def yyy() = 2+1";
    let tokens = Token::tokens_spanned_stream(source);

    let mut parser = front::syntax::parser::CalamarsParser::new(FileId::from(0), tokens);
    let item = parser.parse_item();

    assert!(
        parser.diag().is_empty(),
        "There shuold be no errors parsing"
    );

    let f = match item {
        Item::Declaration(Declaration::Function(fd)) => fd,
        _ => panic!("We should have parsed a function"),
    };

    let body = f.body();

    assert!(
        matches!(body, Expression::BinaryOp(_)),
        "Expression assigned should be a binary op"
    );
}

#[test]
fn parse_fn_with_spaces() {
    let source = "def yyy() = 2 + 1";
    let tokens = Token::tokens_spanned_stream(source);

    let mut parser = front::syntax::parser::CalamarsParser::new(FileId::from(0), tokens);
    let item = parser.parse_item();

    assert!(parser.is_finished(), "we shuold have consumed every token");
    assert!(
        parser.diag().is_empty(),
        "There should be no errors parsing"
    );

    let f = match item {
        Item::Declaration(Declaration::Function(fd)) => fd,
        _ => panic!("We should have parsed a function"),
    };

    let body = f.body();

    assert!(
        matches!(body, Expression::BinaryOp(_)),
        "Expression assigned should be a binary op"
    );
}

#[test]
fn parse_unary() {
    let sources = ["def yyy() = -1", "def yyy() = +1"];
    for source in sources {
        let tokens = Token::tokens_spanned_stream(source);

        let mut parser = front::syntax::parser::CalamarsParser::new(FileId::from(0), tokens);
        let item = parser.parse_item();

        assert!(parser.is_finished(), "we shuold have consumed every token");
        println!("{:?}", parser.diag());
        assert!(
            parser.diag().is_empty(),
            "There should be no errors parsing"
        );

        let f = match item {
            Item::Declaration(Declaration::Function(fd)) => fd,
            _ => panic!("We should have parsed a function"),
        };

        let body = f.body();

        assert!(
            matches!(body, Expression::UnaryOp(_)),
            "Expression assigned should be a binary op"
        );
    }
}

#[test]
fn parse_block() {
    let source = "def foo(): Int = { var x: Int = 2;\nx }";
    let tokens = Token::tokens_spanned_stream(source);
    let mut parser = front::syntax::parser::CalamarsParser::new(FileId::from(0), tokens);
    let item = parser.parse_item();
    assert!(
        parser.diag().is_empty(),
        "There shuold have been no parsing errors"
    );

    let dec = match item {
        Item::Declaration(Declaration::Function(f)) => f,
        _ => panic!("We shuold have parsed a func declaration"),
    };

    assert!(
        matches!(dec.body(), Expression::Block(_)),
        "Function body shuold be a block"
    );
}
