use chumsky::span::SimpleSpan;

pub type Span = SimpleSpan;

#[cfg(test)]
mod test_span {
    use chumsky::{Parser, span::SimpleSpan};

    use crate::{
        parser::{TokenInput, declaration::parse_cldeclaration, parse_cl_item},
        syntax::{
            ast::{ClCompoundExpression, ClDeclaration, ClExpression, ClItem, IfStm},
            token::Token,
        },
    };

    fn stream_for<'a>(src: &'a str) -> impl TokenInput<'a> {
        Token::tokens_spanned_stream(src)
    }

    #[test]
    pub fn identifier_span() {
        let source = "var name: Int = 2;";
        let stream = stream_for(source);
        let (out, errors) = parse_cl_item().parse(stream).into_output_errors();

        assert!(out.is_some());
        let dec = match out.unwrap() {
            ClItem::Declaration(ClDeclaration::Binding(dec)) => dec,
            _ => panic!("This should be a declaration"),
        };

        // Test the span of the name
        assert_eq!(
            dec.name_span(),
            SimpleSpan {
                start: 4,
                end: 8,
                context: ()
            }
        );

        assert_eq!(
            dec.span(),
            SimpleSpan {
                start: 0,
                end: source.len(),
                context: ()
            }
        );

        assert_eq!(
            dec.type_span().unwrap(),
            SimpleSpan {
                start: 10,
                end: 13,
                context: ()
            }
        )
    }

    #[test]
    pub fn if_stm_span() {
        let source = "if true then 2 else 4";
        let stream = stream_for(source);
        let (out, errors) = parse_cl_item().parse(stream).into_output_errors();

        assert!(out.is_some());
        let ifstm = match out.unwrap() {
            ClItem::Expression(ClExpression::IfStm(i)) => i,
            _ => panic!("This shuold be an if statment"),
        };

        assert_eq!(
            ifstm.span(),
            SimpleSpan {
                start: 0,
                end: source.len(),
                context: ()
            }
        );

        assert_eq!(
            ifstm.pred_span(),
            SimpleSpan {
                start: 3,
                end: 7,
                context: ()
            }
        );

        assert_eq!(
            ifstm.then_span(),
            SimpleSpan {
                start: 13,
                end: 14,
                context: ()
            }
        );

        assert_eq!(
            ifstm.else_span(),
            SimpleSpan {
                start: 20,
                end: 21,
                context: ()
            }
        );
    }

    #[test]
    pub fn block_expr() {
        let source = "{ var x: int = 2; 2 } ";
        let stream = stream_for(source);
        let (out, errors) = parse_cl_item().parse(stream).into_output_errors();

        assert!(out.is_some());
        let out = out.unwrap();
        assert_eq!(
            out.span(),
            SimpleSpan {
                start: 0,
                end: source.len() - 1,
                context: ()
            }
        );

        let block = match out.get_exp() {
            ClExpression::Block(block) => block,
            _ => panic!("This sould be a compound exp"),
        };

        let dec = match block.items[0].get_dec() {
            ClDeclaration::Binding(cl_binding) => cl_binding,
            ClDeclaration::Function(cl_func_dec) => panic!("This sould be a varible declaration"),
        };

        assert_eq!(
            dec.name_span(),
            SimpleSpan {
                start: 6,
                end: 7,
                context: ()
            }
        );

        assert_eq!(
            dec.type_span().unwrap(),
            SimpleSpan {
                start: 9,
                end: 12,
                context: ()
            }
        );
    }
}
