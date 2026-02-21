//! A parser for Calamars files!

use calamars_core::ids;

use crate::syntax::{
    ast::{self, Apply, Ident},
    errors::ParsingError,
    span::Span,
    token::Token,
};

fn uni_span(s: &Span) -> Span {
    Span::from(s.start..s.start)
}

const UNARY_BP: usize = 90;
const FUN_CALL: usize = 80;

// Precedence information
const fn precedence(b: &Token) -> Option<(usize, usize)> {
    match b {
        Token::Pow => Some((70, 70)),
        Token::Star | Token::Slash | Token::Mod => Some((60, 61)),
        Token::Plus | Token::Minus | Token::Concat => Some((50, 51)),
        Token::GreaterEqual | Token::LessEqual | Token::Greater | Token::Less => Some((40, 41)),
        Token::EqualEqual | Token::NotEqual => Some((35, 36)),
        Token::And => Some((30, 31)),
        Token::Xor => Some((25, 26)),
        Token::Or => Some((20, 21)),
        _ => None,
    }
}

const fn binary_op_from_token(token: &Token) -> Option<ast::BinaryOperator> {
    match token {
        Token::Or => Some(ast::BinaryOperator::Or),
        Token::And => Some(ast::BinaryOperator::And),
        Token::Xor => Some(ast::BinaryOperator::Xor),
        Token::EqualEqual => Some(ast::BinaryOperator::EqEq),
        Token::NotEqual => Some(ast::BinaryOperator::NotEqual),
        Token::LessEqual => Some(ast::BinaryOperator::Leq),
        Token::GreaterEqual => Some(ast::BinaryOperator::Geq),
        Token::Concat => Some(ast::BinaryOperator::Concat),
        Token::Plus => Some(ast::BinaryOperator::Add),
        Token::Minus => Some(ast::BinaryOperator::Sub),
        Token::Star => Some(ast::BinaryOperator::Times),
        Token::Pow => Some(ast::BinaryOperator::Pow),
        Token::Slash => Some(ast::BinaryOperator::Div),
        Token::Less => Some(ast::BinaryOperator::Less),
        Token::Greater => Some(ast::BinaryOperator::Greater),
        Token::Mod => Some(ast::BinaryOperator::Mod),

        // Not a binary operator!
        _ => None,
    }
}

pub struct CalamarsParser {
    /// Id for the file for which this parser is responsible
    fileid: ids::FileId,
    // Input tokens
    tokens: Vec<(Token, Span)>,
    /// Keep a set of errors that need to be reported
    diag: Vec<ParsingError>,
    curr_index: usize,
}

impl CalamarsParser {
    const ERR_IDENT: &'static str = "<err_ident>";

    /// This function guarantees:
    /// - Tokens is never empty
    pub fn new(fileid: ids::FileId, tokens: Vec<(Token, logos::Span)>) -> Self {
        let mut tokens = tokens
            .into_iter()
            .map(|(t, s)| (t, Span::from(s)))
            .collect::<Vec<_>>();

        // If it does not end on an end of file token, then we add it
        if !matches!(tokens.last().map(|x| &x.0), Some(Token::EOF)) {
            let end = tokens.last().map(|(_, s)| s.end).unwrap_or_default();
            tokens.push((Token::EOF, Span::from(end..end)));
        }

        Self {
            fileid,
            tokens,
            diag: vec![],
            curr_index: 0,
        }
    }

    fn insert_err(&mut self, err: ParsingError) {
        self.diag.push(err);
    }

    fn expect_err(&mut self, expect: impl Into<String>) {
        self.insert_err(ParsingError::Expected {
            expected: expect.into(),
            span: self.zero_width_here(),
        });
    }

    /// Get the index of the EOF token
    fn eof_index(&self) -> usize {
        self.tokens.len().saturating_sub(1)
    }

    fn at_end(&self) -> bool {
        self.n_index() == self.eof_index()
    }

    /// Use before consuming the first token
    fn begin_span(&self) -> usize {
        self.next_span().start
    }

    /// Use after consuiming the last token
    fn end_span(&self) -> usize {
        self.last_consumed_span().end
    }

    fn zero_width_here(&self) -> Span {
        let span = self.next_ref().1;
        Span::from(span.start..span.start)
    }

    fn last_consumed_span(&self) -> Span {
        self.tokens[self.n_index().saturating_sub(1)].1
    }

    fn next_span(&self) -> Span {
        self.tokens[self.n_index()].1
    }

    /// Get the index of the next non-consumed token
    ///
    /// We are guaranteed that this index will be in range for tokens indexing
    fn n_index(&self) -> usize {
        self.eof_index().min(self.curr_index)
    }

    /// Increase the current index by one
    fn advance_one(&mut self) {
        self.curr_index += 1;
    }

    /// The next token needs to be `token`, if it isn't then `err` will be added to the diagnostics.
    fn need(&mut self, token: Token, err: &str) {
        if self.next_eq(token) {
            self.advance_one();
        } else {
            self.expect_err(err);
        }
    }

    /// Get the next token by reference
    fn next_token_ref(&self) -> &Token {
        &self.tokens[self.n_index()].0
    }

    fn next_ref(&self) -> &(Token, Span) {
        &self.tokens[self.n_index()]
    }

    fn next_if_matches(&self, f: impl FnOnce(&Token) -> bool) -> Option<&(Token, Span)> {
        if self.next_matches(f) {
            Some(self.next_ref())
        } else {
            None
        }
    }

    /// Check if the next token meets some predicate
    fn next_matches(&self, f: impl FnOnce(&Token) -> bool) -> bool {
        let next_token = self.next_token_ref();
        f(next_token)
    }

    /// Check if we have finished parsing all of the source tokens
    pub fn is_finished(&self) -> bool {
        self.next_matches(|tk| *tk == Token::EOF)
    }

    fn next_eq(&self, tk: Token) -> bool {
        tk == *self.next_token_ref()
    }

    fn checkpoint(&self) -> (usize, usize) {
        (self.curr_index, self.diag.len())
    }

    fn rollback(&mut self, (index, diagl): (usize, usize)) {
        self.curr_index = index;
        self.diag.truncate(diagl);
    }

    /// Keep skipping tokens until some predicate is met.
    ///
    /// This can be used for error recovery.
    fn skip_until<Ctx>(&mut self, mut init: Ctx, until: impl Fn(&mut Ctx, &Token) -> bool) {
        loop {
            let next_token = self.next_token_ref();
            if until(&mut init, next_token) || self.at_end() {
                return;
            }

            self.advance_one();
        }
    }

    /// Parse a semicolon. If one was not found, we will log an error, and "pretend" as semicolon
    /// was here as a form of recovery
    fn handle_semicolon(&mut self) {
        if self.next_eq(Token::Semicolon) {
            self.advance_one();
        } else {
            self.expect_err(";");
        }
    }

    fn parse_identifier(&mut self) -> ast::Ident {
        match self.next_ref().clone() {
            (Token::Ident(id), span) => {
                self.advance_one();
                ast::Ident::new(id, span)
            }
            (_, span) => {
                let unit_span = Span::from(span.start..span.start);
                self.insert_err(ParsingError::Expected {
                    expected: "identifier".to_string(),
                    span: unit_span,
                });
                // For now, usng <err_ident> is ok, since we dont allow idents to start with '<' in
                // the tokenizer. This is not great, but it will be fixed when we implement string
                // interning.
                ast::Ident::new(Self::ERR_IDENT.into(), unit_span)
            }
        }
    }

    /// Try to parse a literal.
    fn try_lit(&self) -> Option<ast::Literal> {
        let (tk, span) = self.next_ref();
        let lit_kind = match tk {
            Token::True => ast::LiteralKind::Boolean(true),
            Token::False => ast::LiteralKind::Boolean(false),
            Token::Int(i) => ast::LiteralKind::Integer(*i),
            Token::Float(f) => ast::LiteralKind::Real(*f),
            Token::Char(c) => ast::LiteralKind::Char(*c),
            Token::String(s) => ast::LiteralKind::String(s.clone()),
            _ => return None,
        };
        Some(ast::Literal::new(lit_kind, *span))
    }

    /// if <expr> then <expr> else <expr>  
    fn parse_if_expr(&mut self) -> ast::Expression {
        let start = self.begin_span();

        self.need(Token::If, "if");
        let pred = self.parse_expression();

        self.need(Token::Then, "then");
        let then = self.parse_expression();

        self.need(Token::Else, "else");

        let otherwise = self.parse_expression();
        let end = self.end_span();
        let ifs = ast::IfStm::new(
            pred.into(),
            then.into(),
            otherwise.into(),
            Span::from(start..end),
        );
        ast::Expression::IfStm(ifs)
    }

    fn parse_block_expr(&mut self) -> ast::Expression {
        let start = self.begin_span();
        let opening_loc = self.zero_width_here();
        self.need(Token::LBrace, "{");
        let mut items = vec![];
        let mut final_expr = None;

        loop {
            match self.next_token_ref() {
                Token::TypeDef | Token::Def | Token::DocComment(_) => {
                    let d = self.parse_declaration();
                    items.push(ast::Item::Declaration(d));
                }
                Token::Semicolon => {
                    // TODO: Warning, semicolon not necessary
                    self.advance_one();
                }
                tk if self.is_expr_init(tk) => {
                    let e = self.parse_expression();
                    // If this is the tail of the function, we will return it
                    if self.next_eq(Token::RBrace) {
                        final_expr = Some(Box::new(e));
                        break;
                    } else {
                        items.push(ast::Item::Expression(e));
                        self.handle_semicolon();
                    }
                }
                Token::RBrace => break,
                _ => {
                    // TODO:Skip until a good point here, recover
                    break;
                }
            }
        }

        if self.next_eq(Token::RBrace) {
            self.advance_one();
        } else {
            self.diag.push(ParsingError::DelimeterNotClosed {
                expected: "}",
                at: self.zero_width_here(),
                opening_loc,
            });
        }

        let end = self.end_span();
        let comp = ast::CompoundExpression::new(items, final_expr, Span::from(start..end));
        ast::Expression::Block(comp)
    }

    /// Literals, name/path, bracketed expressions
    fn parse_primary(&mut self) -> ast::Expression {
        // A literal is the simplest case
        if let Some(literal) = self.try_lit() {
            self.advance_one();
            return ast::Expression::Literal(literal);
        }
        let (tk, sp) = self.next_ref().clone();
        match tk {
            Token::Ident(_) => ast::Expression::Identifier(self.parse_identifier()),
            Token::LParen => {
                self.advance_one();
                let expr = self.parse_expression();

                // Try to find closing paren
                // If we dont find it, we will artificially insert it
                if !self.next_matches(|t| *t == Token::RParen) {
                    self.insert_err(ParsingError::DelimeterNotClosed {
                        opening_loc: sp,
                        expected: ")",
                        at: self.zero_width_here(),
                    });
                } else {
                    self.advance_one();
                }

                expr
            }
            Token::LBrace => self.parse_block_expr(),
            Token::If => self.parse_if_expr(),
            _ => {
                let at = self.zero_width_here();
                self.insert_err(ParsingError::Expected {
                    expected: "expression".into(),
                    span: at,
                });
                ast::Expression::Error(at)
            }
        }
    }

    /// Parse comma separated arguments.
    ///
    /// To call this, you must ensure that the first token is LParen.
    fn parse_fn_arguments(&mut self) -> Vec<ast::Expression> {
        let mut arguments = vec![];
        self.advance_one();

        loop {
            // Handle the case where there are no arguments
            if self.next_matches(|tk| *tk == Token::RParen) {
                self.advance_one();
                return arguments;
            }

            arguments.push(self.parse_expression());

            if self.next_matches(|tk| *tk == Token::Comma) {
                self.advance_one(); // Consume the comma
            } else {
                break;
            }
        }

        if self.next_matches(|tk| *tk == Token::RParen) {
            self.advance_one();
        } else {
            self.expect_err(")");
        }

        arguments
    }

    fn is_apply_arg_start(&self, token: &Token) -> bool {
        matches!(
            token,
            Token::Ident(_)
                | Token::Int(_)
                | Token::Float(_)
                | Token::True
                | Token::False
                | Token::String(_)
                | Token::Char(_)
                | Token::LParen
                | Token::LBrace
                | Token::If
        )
    }

    /// Given some expression, apply postfix operations to it.
    ///
    /// This will handle application such as `f x` and `f (x)`.
    fn parse_postfix_operation(&mut self, mut expr: ast::Expression) -> ast::Expression {
        loop {
            let token = self.next_token_ref();
            if !self.is_apply_arg_start(token) {
                break;
            }

            let arg = self.parse_primary();
            let span = Span::from(expr.span().start..arg.span().end);
            expr = ast::Expression::Apply(Apply::new(expr, arg, span));
        }
        expr
    }

    fn parse_prefix(&mut self) -> ast::Expression {
        if self.next_eq(Token::Minus) | self.next_eq(Token::Plus) {
            let unary_span = self.next_ref().1;
            self.advance_one();
            let expression = self.parse_expression_with_binding_power(UNARY_BP);
            let span = Span::from(unary_span.start..expression.span().end);
            let unary = ast::UnaryOp::new(ast::UnaryOperator::Neg, expression.into(), span);
            ast::Expression::UnaryOp(unary)
        } else {
            self.parse_primary()
        }
    }

    /// Pratt parsing for binary expressions
    ///
    /// `Expression <symbol> Expression` where <symbol> can be one of the following:
    /// <, <=, >, >=, |>, and, or, xor, *, ^, /, +, -
    fn parse_expression_with_binding_power(&mut self, binding_power: usize) -> ast::Expression {
        let mut lhs = self.parse_prefix();
        lhs = self.parse_postfix_operation(lhs);
        loop {
            let token = self.next_token_ref();
            let operator = precedence(token);
            if operator.is_none() {
                break;
            }

            let (lbp, rbp) = operator.unwrap();
            let operator = binary_op_from_token(token).unwrap();
            if lbp < binding_power {
                break;
            }

            // Consume the operator
            self.advance_one();
            let rhs = self.parse_expression_with_binding_power(rbp);
            let rhs = self.parse_postfix_operation(rhs);
            let span = Span::from(lhs.span().start..rhs.span().end);
            let binop = ast::BinaryOp::new(operator, lhs.into(), rhs.into(), span);
            lhs = ast::Expression::BinaryOp(binop);
        }
        lhs
    }

    fn parse_expression(&mut self) -> ast::Expression {
        self.parse_expression_with_binding_power(0)
    }

    /// Identifiers with a dot separator, for example: `foo.Bar`
    fn parse_path_type(&mut self) -> ast::Type {
        let mut segments = vec![];
        let init_span = self.next_span().start;
        let mut end_span = self.next_span().end;

        loop {
            let ident = self.parse_identifier();
            segments.push(ident.clone());
            end_span = ident.span().end;
            if ident.ident() == Self::ERR_IDENT {
                self.skip_until((), |_, tk| *tk == Token::Equal);
                break;
            }

            // Now find the dot
            if self.next_matches(|tk| *tk == Token::Dot) {
                self.advance_one();
            } else {
                break;
            }
        }

        ast::Type::new_path(segments, Span::from(init_span..end_span))
    }

    /// Parse an array type, this is:
    ///
    /// `[ <type> ]`
    fn parse_arr_type(&mut self) -> ast::Type {
        let opening_loc = self.zero_width_here();
        self.need(Token::LBracket, "[");

        let elem = self.parse_type();

        // expect ']'
        let r_sp = if self.next_matches(|t| *t == Token::RBracket) {
            let sp = self.next_ref().1;
            self.advance_one();
            sp
        } else {
            self.insert_err(ParsingError::DelimeterNotClosed {
                opening_loc,
                expected: "]",
                at: self.zero_width_here(),
            });
            // try to sync
            self.skip_until((), |_, tk| matches!(tk, Token::RBracket | Token::EOF));
            if self.next_matches(|t| *t == Token::RBracket) {
                let sp = self.next_ref().1;
                self.advance_one();
                sp
            } else {
                elem.span()
            }
        };

        ast::Type::Array {
            elem_type: Box::new(elem),
            span: Span::from(opening_loc.start..r_sp.end),
        }
    }

    /// Parse types:
    ///
    /// Possible types currently implemented are:
    /// - Path (something like `String`, or `foo.Bar`)
    /// - Function (`Type -> Type`)
    /// - Array (`[ Type ]`)
    fn parse_type(&mut self) -> ast::Type {
        let span_init = self.begin_span();
        let lhs_type = match self.next_token_ref() {
            Token::Ident(_) => self.parse_path_type(),
            Token::LParen => {
                self.advance_one();
                let inner_type = self.parse_type();
                self.need(Token::RParen, ")");
                inner_type
            }
            Token::LBracket => self.parse_arr_type(),
            _ => {
                self.expect_err("type");
                ast::Type::Error(Span::from(self.end_span()..self.end_span()))
            }
        };

        if !self.next_matches(|tk| matches!(tk, Token::Arrow)) {
            return lhs_type;
        }

        self.advance_one();
        let rhs_type = self.parse_type();
        let span_end = self.end_span();
        ast::Type::Func {
            input: lhs_type.into(),
            output: rhs_type.into(),
            span: Span::from(span_init..span_end),
        }
    }

    fn parse_comma_separated_idents(&mut self) -> Vec<(ast::Ident, ast::Type)> {
        let mut v = vec![];
        loop {
            // We need to make sure that there is a token before parsing ident
            if !self.next_matches(|tk| matches!(tk, Token::Ident(_))) {
                break;
            }
            let ident = self.parse_identifier();

            self.need(Token::Colon, ":");

            let ty = self.parse_type();
            v.push((ident, ty));

            if !self.next_eq(Token::Comma) {
                break;
            }
            self.advance_one();
        }
        v
    }

    /// Parse the type definition of a  declaration
    ///
    /// Examples are:
    /// ```cm
    /// x :: Int
    ///
    /// identity :: Int -> Int
    /// ```
    fn parse_declaration_type(&mut self) -> ast::Declaration {
        #[rustfmt::skip]
        let docs = if let Token::DocComment(comment) = self.next_token_ref() {
            Some(comment.clone())
        } else { None };

        if docs.is_some() {
            self.advance_one();
        }

        self.need(Token::TypeDef, "typ");
        let name = self.parse_identifier();
        self.need(Token::DoubleColon, "::");
        let dtype = self.parse_type();
        ast::Declaration::TypeSignature { docs, name, dtype }
    }

    /// Parse the body of a declaration. This can be a function definition or a variable.
    ///
    /// Examples are:
    /// ```cm
    /// x = 2
    ///
    /// identity x = x
    /// ```
    fn parse_declaration_body(&mut self) -> ast::Declaration {
        self.need(Token::Def, "def");
        let name = self.parse_identifier();
        let mut params = vec![];
        loop {
            match self.next_ref() {
                (Token::Ident(ident), span) => {
                    params.push(Ident::new(ident.clone(), *span));
                }
                _ => break,
            }
            self.advance_one();
        }

        self.need(Token::Equal, "=");
        let body = self.parse_expression();
        ast::Declaration::Binding { name, params, body }
    }

    fn parse_declaration(&mut self) -> ast::Declaration {
        if self.next_eq(Token::Def) {
            self.parse_declaration_body()
        } else {
            self.parse_declaration_type()
        }
    }

    fn is_expr_init(&self, token: &Token) -> bool {
        matches!(
            token,
            Token::Ident(_)
                | Token::Int(_)
                | Token::Float(_)
                | Token::True
                | Token::False
                | Token::String(_)
                | Token::Char(_)
                | Token::LBrace
                | Token::LParen
                | Token::If
                // Unary operators
                | Token::Minus
                | Token::Plus
        )
    }

    pub fn parse_item(&mut self) -> ast::Item {
        match self.next_token_ref() {
            Token::Ident(_) | Token::DocComment(_) => {
                let dec = self.parse_declaration();
                ast::Item::Declaration(dec)
            }
            tk if self.is_expr_init(tk) => ast::Item::Expression(self.parse_expression()),

            _ => panic!("Only items can be parsed here;"),
        }
    }

    pub fn parse_file(&mut self) -> ast::Module {
        let mut decs = vec![];
        loop {
            // Skip until we find a declaration or EOF
            self.skip_until((), |_, tk| {
                matches!(
                    tk,
                    Token::Def
                        | Token::TypeDef
                        | Token::Ident(_)
                        | Token::DocComment(_)
                        | Token::EOF
                )
            });

            if self.next_eq(Token::EOF) {
                break;
            }

            let dec = self.parse_declaration();
            decs.push(dec);
        }

        ast::Module {
            imports: vec![],
            items: decs,
        }
    }

    pub fn diag(&self) -> &[ParsingError] {
        &self.diag
    }
}
