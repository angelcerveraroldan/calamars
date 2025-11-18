//! A parser for Calamars files!

use calamars_core::ids;

use crate::syntax::{
    ast::{self, FuncCall},
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
        Token::Star | Token::Slash => Some((60, 61)),
        Token::Plus | Token::Minus | Token::Concat => Some((50, 51)),
        Token::GreaterEqual | Token::LessEqual => Some((40, 41)),
        Token::EqualEqual | Token::NotEqual => Some((35, 36)),
        Token::And => Some((30, 31)),
        Token::Xor => Some((25, 26)),
        Token::Or => Some((20, 21)),
        _ => None,
    }
}

const fn binary_op_from_token(token: &Token) -> Option<ast::BinaryOperator> {
    match token {
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
    /// We are guaranteed that this insdex will be in range for tokens indexing
    fn n_index(&self) -> usize {
        self.eof_index().min(self.curr_index)
    }

    /// Increase the current index by one
    fn advance_one(&mut self) {
        self.curr_index += 1;
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

    fn next_eq(&self, tk: Token) -> bool {
        tk == *self.next_token_ref()
    }

    /// Get the next n tokens
    fn peek_n(&self, n: usize) -> Box<[&Token]> {
        let next = self.n_index();
        let range = next..n.min(next);
        self.tokens[range]
            .iter()
            .map(|(a, _)| a)
            .collect::<Box<[&Token]>>()
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

    // Literals, name/path, bracketed expressions
    fn parse_primary(&mut self) -> ast::Expression {
        // A literal is the simplest case
        if let Some(literal) = self.try_lit() {
            self.advance_one();
            return ast::Expression::Literal(literal);
        }
        let (tk, sp) = self.next_ref().clone();
        match tk {
            // TODO: For now we just supprt idents to be one word
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
                }

                expr
            }
            Token::LBrace => {
                let start = self.begin_span();
                self.advance_one();
                let mut items = vec![];
                let mut final_expr = None;

                loop {
                    match self.next_token_ref() {
                        Token::Var | Token::Val | Token::Def => {
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
                    // We need wayy better errors, at least be consistent.
                    self.expect_err("closing brace '}'");
                }

                let end = self.end_span();
                let comp = ast::CompoundExpression::new(items, final_expr, Span::from(start..end));
                ast::Expression::Block(comp)
            }
            Token::If => {
                let start = self.begin_span();
                self.advance_one();
                let pred = self.parse_expression();

                if !self.next_eq(Token::Then) {
                    self.expect_err("then");
                } else {
                    self.advance_one();
                }
                let then = self.parse_expression();

                if !self.next_eq(Token::Else) {
                    self.expect_err("else");
                } else {
                    self.advance_one();
                }

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

    /// Given some expression, apply postfix operations to it.
    ///
    /// This will handle things such as `f(x)`. We will apply `(x)` to `f`.
    fn parse_postfix_operation(&mut self, mut expr: ast::Expression) -> ast::Expression {
        loop {
            let (tk, sp) = self.next_ref().clone();
            match tk {
                // Apply parameters to input
                Token::LParen => {
                    let args = self.parse_fn_arguments();
                    let span_end = self.last_consumed_span();
                    let span = Span::from(sp.start..span_end.end);
                    expr = ast::Expression::FunctionCall(FuncCall::new(expr, args, span));
                }
                // TODO: Index not yet supported
                Token::LBracket => {
                    self.skip_until(1, |a, b| {
                        if matches!(b, Token::RBracket) {
                            *a -= 1;
                        }
                        if matches!(b, Token::LBracket) {
                            *a += 1;
                        }
                        *a == 0
                    });
                }
                _ => break,
            }
        }
        expr
    }

    fn parse_prefix(&mut self) -> ast::Expression {
        if self.next_matches(|t| *t == Token::Minus) {
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

    fn parse_fn_type(&mut self) -> ast::Type {
        let init_span = self.next_span();
        // Start by consuming the first paren
        if !self.next_matches(|tk| *tk == Token::LParen) {
            self.expect_err("function type");
            return ast::Type::Error;
        } else {
            self.advance_one();
        }

        let mut inputs = vec![];
        loop {
            if *self.next_token_ref() == Token::RParen {
                self.advance_one();
                break;
            }

            let ty = self.parse_type();
            inputs.push(ty);

            if *self.next_token_ref() == Token::Comma {
                self.advance_one();
            }
        }

        if !self.next_matches(|tk| *tk == Token::Arrow) {
            self.expect_err("->");
        } else {
            self.advance_one();
        }

        let output = Box::new(self.parse_type());
        let out_final = output.span().unwrap().end;
        ast::Type::Func {
            inputs,
            output,
            span: Some(Span::from(init_span.start..out_final)),
        }
    }

    fn parse_arr_type(&mut self) -> ast::Type {
        let l_sp = self.next_ref().1;
        self.advance_one(); // '['

        let elem = self.parse_type();

        // expect ']'
        let r_sp = if self.next_matches(|t| *t == Token::RBracket) {
            let sp = self.next_ref().1;
            self.advance_one();
            sp
        } else {
            self.insert_err(ParsingError::DelimeterNotClosed {
                opening_loc: l_sp,
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
                elem.span().unwrap_or(l_sp)
            }
        };

        ast::Type::Array {
            elem_type: Box::new(elem),
            span: Span::from(l_sp.start..r_sp.end),
        }
    }

    /// Parse types
    ///
    /// Possible types currently implemented are:
    /// - Path (something like `String`, or `foo.Bar`)
    /// - Function (`Type -> Type`)
    /// - Array (`[ Type ]`)
    fn parse_type(&mut self) -> ast::Type {
        match self.next_token_ref() {
            Token::Ident(_) => self.parse_path_type(),
            Token::LParen => self.parse_fn_type(),
            Token::LBracket => self.parse_arr_type(),
            _ => {
                self.expect_err("type");
                ast::Type::Error
            }
        }
    }

    /// Parse bindings such as
    ///
    /// ```cm
    /// var x: Int = expression;
    /// val y: Int = expression;
    /// ```
    /// To call this function, you need to assure that the next token is either Var or Val
    fn parse_binding(&mut self) -> ast::Binding {
        let start = self.begin_span();

        let mutable = if self.next_eq(Token::Var) {
            true
        } else if self.next_eq(Token::Val) {
            false
        } else {
            panic!("This function should only be executed if the first token is var or val");
        };

        self.advance_one();
        let name = self.parse_identifier();
        if self.next_eq(Token::Colon) {
            self.advance_one();
        } else {
            self.expect_err(":");
        }
        let vtype = self.parse_type();

        if self.next_eq(Token::Equal) {
            self.advance_one();
        } else {
            self.expect_err("=");
        }

        let assigned = Box::new(self.parse_expression());
        self.handle_semicolon();

        let end = self.end_span();
        ast::Binding::new(name, vtype, assigned, mutable, Span::from(start..end))
    }

    fn parse_comma_separated_idents(&mut self) -> Vec<(ast::Ident, ast::Type)> {
        let mut v = vec![];
        loop {
            // We need to make sure that ther is a token before parsing ident
            if !self.next_matches(|tk| matches!(tk, Token::Ident(_))) {
                break;
            }
            let ident = self.parse_identifier();

            if self.next_eq(Token::Colon) {
                self.advance_one();
            } else {
                self.expect_err(":");
            }

            let ty = self.parse_type();
            v.push((ident, ty));

            if !self.next_eq(Token::Comma) {
                break;
            }
            self.advance_one();
        }
        v
    }

    /// Parse a function declaration such as
    ///
    /// ```cm
    /// def main() = {
    ///   var x: Int = 2;
    /// }
    /// ```
    /// To call this function, you need to assure that the next token is Def
    fn parse_function_declaration(&mut self) -> ast::FuncDec {
        // There may be a doc-comment
        let next_token = self.next_token_ref();
        let doc_comment = if let Token::DocComment(comment) = next_token {
            Some(comment.clone())
        } else {
            None
        };

        if doc_comment.is_some() {
            self.advance_one();
        }

        let start = self.begin_span();
        if !self.next_eq(Token::Def) {
            panic!("First token must be Def");
        }

        self.advance_one();
        let fname = self.parse_identifier();

        if !self.next_eq(Token::LParen) {
            self.expect_err("(");
        } else {
            self.advance_one();
        }

        let params = self.parse_comma_separated_idents();

        if self.next_eq(Token::RParen) {
            self.advance_one();
        } else {
            self.expect_err(")");

            // Skip until we are out of the input brackets.
            self.skip_until(1, |a, b| {
                if matches!(b, Token::RParen) {
                    *a -= 1;
                }
                if matches!(b, Token::LParen) {
                    *a += 1;
                }
                *a == 0
            });
        }

        let out_ty = if self.next_eq(Token::Colon) {
            self.advance_one();
            self.parse_type()
        } else {
            ast::Type::Unit
        };

        if self.next_eq(Token::Equal) {
            self.advance_one();
        } else {
            self.expect_err("=");
        }

        let expr = self.parse_expression();
        let end = self.end_span();
        let total_span = Span::from(start..end);
        ast::FuncDec::new(fname, params, out_ty, expr, total_span, doc_comment)
    }

    fn parse_declaration(&mut self) -> ast::Declaration {
        match self.next_token_ref() {
            Token::Def => ast::Declaration::Function(self.parse_function_declaration()),
            Token::Var | Token::Val => ast::Declaration::Binding(self.parse_binding()),
            _ => panic!("not a declaration"),
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
        )
    }

    pub fn parse_item(&mut self) -> ast::Item {
        match self.next_token_ref() {
            Token::Def | Token::Var | Token::Val => {
                ast::Item::Declaration(self.parse_declaration())
            }
            tk if self.is_expr_init(tk) => ast::Item::Expression(self.parse_expression()),

            _ => panic!("Only items can be parsed here;"),
        }
    }

    pub fn parse_file(&mut self) -> ast::Module {
        let mut decs = vec![];
        loop {
            // Skip until we find a def, var, or val (only declarations are supported on top level,
            // later also imports)
            self.skip_until((), |_, tk| {
                matches!(tk, Token::Def | Token::Var | Token::Val)
            });

            // If we finished the file without finding the above tokens, we exit
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

#[cfg(test)]
mod tests {

    use super::*;

    fn sp(start: usize, end: usize) -> logos::Span {
        logos::Span { start, end }
    }

    fn toks(pairs: &[(Token, (usize, usize))]) -> Vec<(Token, logos::Span)> {
        pairs
            .iter()
            .map(|(t, (s, e))| (t.clone(), sp(*s, *e)))
            .collect()
    }

    fn parse_expr_from_tokens(
        tokens: Vec<(Token, logos::Span)>,
    ) -> (CalamarsParser, ast::Expression) {
        let file = ids::FileId::from(0);
        let mut p = CalamarsParser::new(file, tokens);
        let expr = p.parse_expression();
        (p, expr)
    }

    fn parse_ty(tokens: Vec<(Token, logos::Span)>) -> (CalamarsParser, ast::Type) {
        let file = ids::FileId::from(0);
        let mut p = CalamarsParser::new(file, tokens);
        let ty = p.parse_type();
        (p, ty)
    }

    fn parse_varval(tokens: Vec<(Token, logos::Span)>) -> (CalamarsParser, ast::Binding) {
        let file = ids::FileId::from(0);
        let mut p = CalamarsParser::new(file, tokens);
        let ty = p.parse_binding();
        (p, ty)
    }

    fn parse_fn(tokens: Vec<(Token, logos::Span)>) -> (CalamarsParser, ast::FuncDec) {
        let file = ids::FileId::from(0);
        let mut p = CalamarsParser::new(file, tokens);
        let ty = p.parse_function_declaration();
        (p, ty)
    }

    fn bool(f: usize, t: usize) -> ast::Type {
        ast::Type::Path {
            segments: vec![ast::Ident::new("Bool".into(), Span::from(f..t))],
            span: Span::from(f..t),
        }
    }

    fn int(f: usize, t: usize) -> ast::Type {
        ast::Type::Path {
            segments: vec![ast::Ident::new("Int".into(), Span::from(f..t))],
            span: Span::from(f..t),
        }
    }

    fn float(f: usize, t: usize) -> ast::Type {
        ast::Type::Path {
            segments: vec![ast::Ident::new("Float".into(), Span::from(f..t))],
            span: Span::from(f..t),
        }
    }

    #[test]
    fn expr_precedence_mul_over_add() {
        // 1 + 2 * 3
        let tokens = toks(&[
            (Token::Int(1), (0, 1)),
            (Token::Plus, (2, 3)),
            (Token::Int(2), (4, 5)),
            (Token::Star, (6, 7)),
            (Token::Int(3), (8, 9)),
            (Token::EOF, (9, 9)),
        ]);
        let (p, e) = parse_expr_from_tokens(tokens);

        // 2 * 3
        let mult = ast::Expression::BinaryOp(ast::BinaryOp::new(
            ast::BinaryOperator::Times,
            Box::new(ast::Expression::Literal(ast::Literal::new(
                ast::LiteralKind::Integer(2),
                Span::from(4..5),
            ))),
            Box::new(ast::Expression::Literal(ast::Literal::new(
                ast::LiteralKind::Integer(3),
                Span::from(8..9),
            ))),
            Span::from(4..9),
        ));

        // 1 + mult
        let expected = ast::Expression::BinaryOp(ast::BinaryOp::new(
            ast::BinaryOperator::Add,
            Box::new(ast::Expression::Literal(ast::Literal::new(
                ast::LiteralKind::Integer(1),
                Span::from(0..1),
            ))),
            Box::new(mult),
            Span::from(0..9),
        ));

        assert!(p.diag.is_empty(), "should parse without diagnostics");
        assert!(e == expected);
    }

    #[test]
    fn expr_calls_chain() {
        // f(1)(2 + 3)
        let tokens = toks(&[
            (Token::Ident("f".into()), (0, 1)),
            (Token::LParen, (1, 2)),
            (Token::Int(1), (2, 3)),
            (Token::RParen, (3, 4)),
            (Token::LParen, (4, 5)),
            (Token::Int(2), (5, 6)),
            (Token::Plus, (7, 8)),
            (Token::Int(3), (9, 10)),
            (Token::RParen, (10, 11)),
            (Token::EOF, (11, 11)),
        ]);
        let (p, e) = parse_expr_from_tokens(tokens);

        assert!(matches!(e, ast::Expression::FunctionCall(_)));
        assert!(p.diag.is_empty(), "call chaining should parse cleanly");
    }

    #[test]
    fn expr_grouping_then_mul() {
        // (a + b) * c
        let tokens = toks(&[
            (Token::LParen, (0, 1)),
            (Token::Ident("a".into()), (1, 2)),
            (Token::Plus, (3, 4)),
            (Token::Ident("b".into()), (5, 6)),
            (Token::RParen, (6, 7)),
            (Token::Star, (8, 9)),
            (Token::Ident("c".into()), (10, 11)),
            (Token::EOF, (11, 11)),
        ]);
        let (p, e) = parse_expr_from_tokens(tokens);
        assert!(matches!(e, ast::Expression::BinaryOp(_)));
        match e {
            ast::Expression::BinaryOp(b) => {
                assert!(matches!(b.rhs().as_ref(), ast::Expression::Identifier(_)));
            }
            _ => panic!(),
        }
        assert!(
            p.diag.is_empty(),
            "grouping + precedence should parse without diagnostics"
        );
    }

    #[test]
    fn expr_missing_rparen_in_call_emits_diag() {
        // f(1
        let tokens = toks(&[
            (Token::Ident("f".into()), (0, 1)),
            (Token::LParen, (1, 2)),
            (Token::Int(1), (2, 3)),
            (Token::EOF, (3, 3)),
        ]);
        let (p, _e) = parse_expr_from_tokens(tokens);
        assert!(!p.diag.is_empty(), "should report a missing `)` diagnostic");
    }

    #[test]
    fn expr_missing_rhs_after_plus_emits_diag() {
        // 1 +
        let tokens = toks(&[
            (Token::Int(1), (0, 1)),
            (Token::Plus, (2, 3)),
            (Token::EOF, (3, 3)),
        ]);
        let (mut p, _e) = parse_expr_from_tokens(tokens);
        let _ = p.parse_expression();

        assert!(
            !p.diag.is_empty(),
            "should report an expected expression after `+`"
        );
    }

    #[test]
    fn type_array_parsers_no_error() {
        let tokens = toks(&[
            (Token::LBracket, (0, 1)),
            (Token::Ident("int".into()), (2, 5)),
            (Token::RBracket, (6, 7)),
        ]);

        let (p, ty) = parse_ty(tokens);
        assert!(p.diag.is_empty());
        assert!(matches!(ty, ast::Type::Array { .. }));
    }

    #[test]
    fn type_path_simple() {
        // String
        let tokens = toks(&[
            (Token::Ident("String".into()), (0, 6)),
            (Token::EOF, (6, 6)),
        ]);
        let (p, ty) = parse_ty(tokens);
        assert!(
            p.diag.is_empty(),
            "should parse simple path without diagnostics"
        );
    }

    #[test]
    fn type_fn_two_params() {
        // (Int, Float) -> Bool
        let tokens = toks(&[
            (Token::LParen, (0, 1)),
            (Token::Ident("Int".into()), (1, 4)),
            (Token::Comma, (4, 5)),
            (Token::Ident("Float".into()), (6, 11)),
            (Token::RParen, (11, 12)),
            (Token::Arrow, (13, 15)),
            (Token::Ident("Bool".into()), (16, 20)),
            (Token::EOF, (20, 20)),
        ]);
        let (p, ty) = parse_ty(tokens);
        assert!(
            p.diag.is_empty(),
            "should parse (Int, Float) -> Bool without diagnostics"
        );

        assert!(matches!(ty, ast::Type::Func { .. }));
        match ty {
            ast::Type::Func {
                inputs,
                output,
                span,
            } => {
                assert_eq!(
                    *output.as_ref(),
                    ast::Type::new_path(
                        vec![ast::Ident::new("Bool".into(), Span::from(16..20))],
                        Span::from(16..20)
                    )
                )
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn parse_binding() {
        let tokens = toks(&[
            (Token::Var, (0, 1)),
            (Token::Ident("x".into()), (1, 2)),
            (Token::Colon, (4, 5)),
            (Token::Ident("Float".into()), (6, 11)),
            (Token::Equal, (12, 13)),
            (Token::Int(12), (14, 15)),
            (Token::Semicolon, (16, 17)),
            (Token::EOF, (20, 20)),
        ]);
        let (p, var) = parse_varval(tokens);
        println!("{:?}", p.diag);
        assert!(p.diag.is_empty(), "Parse binding without errors");

        assert!(matches!(var.vtype, ast::Type::Path { .. }));
        if let ast::Type::Path { segments, span } = var.vtype.clone() {
            assert_eq!(segments[0].ident(), "Float");
        }
        assert_eq!(var.vname.ident(), "x");
        assert_eq!(var.span(), Span::from(0..17));
    }

    #[test]
    fn lambda_fn() {
        // val x: (Int, Float) -> Bool = 12;
        //
        // The spans make no sense for this test, but theyre not really beind read.
        let tokens = toks(&[
            (Token::Val, (0, 1)),
            (Token::Ident("x".into()), (1, 2)),
            (Token::Colon, (4, 5)),
            (Token::LParen, (0, 1)),
            (Token::Ident("Int".into()), (1, 4)),
            (Token::Comma, (4, 5)),
            (Token::Ident("Float".into()), (6, 11)),
            (Token::RParen, (11, 12)),
            (Token::Arrow, (13, 15)),
            (Token::Ident("Bool".into()), (16, 20)),
            (Token::Equal, (12, 13)),
            (Token::Int(12), (14, 15)),
            (Token::Semicolon, (16, 17)),
            (Token::EOF, (20, 20)),
        ]);
        let (p, var) = parse_varval(tokens);
        assert!(p.diag.is_empty(), "Parse binding without errors");
        assert!(matches!(var.vtype, ast::Type::Func { .. }));

        let bool = bool(16, 20);
        let int = int(1, 4);
        let float = float(6, 11);

        if let ast::Type::Func { inputs, output, .. } = var.vtype.clone() {
            assert_eq!(output.as_ref(), &bool, "var has type fn with bool output");
            assert_eq!(inputs[0], int, "var has fn type with first input int");
            assert_eq!(inputs[1], float, "var has fn type with second input float");
        }
        assert_eq!(var.vname.ident(), "x");
        assert_eq!(var.span(), Span::from(0..17));
    }

    #[test]
    fn fn_simple_no_params_expr_body() {
        // def cn(): Int = 42
        let tokens = toks(&[
            (Token::Def, (0, 3)),
            (Token::Ident("cn".into()), (4, 6)),
            (Token::LParen, (6, 7)),
            (Token::RParen, (7, 8)),
            (Token::Colon, (9, 10)),
            (Token::Ident("Int".into()), (11, 14)),
            (Token::Equal, (15, 16)),
            (Token::Int(42), (17, 19)),
            (Token::EOF, (19, 19)),
        ]);
        let (p, f) = parse_fn(tokens);
        assert_eq!(f.airity(), 0);
        assert!(
            p.diag.is_empty(),
            "should parse simple function without diagnostics"
        );
    }

    #[test]
    fn fn_two_params_with_types() {
        // def sum(a: Int, b: Bool): Int = a
        let tokens = toks(&[
            (Token::Def, (0, 3)),
            (Token::Ident("sum".into()), (4, 7)),
            (Token::LParen, (7, 8)),
            (Token::Ident("a".into()), (8, 9)),
            (Token::Colon, (9, 10)),
            (Token::Ident("Int".into()), (11, 14)),
            (Token::Comma, (14, 15)),
            (Token::Ident("b".into()), (16, 17)),
            (Token::Colon, (17, 18)),
            (Token::Ident("Bool".into()), (19, 22)),
            (Token::RParen, (22, 23)),
            (Token::Colon, (24, 25)),
            (Token::Ident("Int".into()), (26, 29)),
            (Token::Equal, (30, 31)),
            (Token::Ident("a".into()), (32, 33)),
            (Token::EOF, (33, 33)),
        ]);
        let (p, f) = parse_fn(tokens);
        assert_eq!(f.airity(), 2);
        let ty = f.fntype();
        assert!(matches!(ty, ast::Type::Func { .. }));
        if let ast::Type::Func { inputs, output, .. } = ty {
            assert_eq!(output.as_ref().clone(), int(26, 29));
            assert_eq!(inputs[0], int(11, 14));
            assert_eq!(inputs[1], bool(19, 22));
        }
        assert!(
            p.diag.is_empty(),
            "params + return type should parse cleanly"
        );
    }

    #[test]
    fn fn_trailing_comma_in_params_is_ok() {
        // def f(a: Int,): Int = a
        let tokens = toks(&[
            (Token::Def, (0, 3)),
            (Token::Ident("f".into()), (4, 5)),
            (Token::LParen, (5, 6)),
            (Token::Ident("a".into()), (6, 7)),
            (Token::Colon, (7, 8)),
            (Token::Ident("Int".into()), (9, 12)),
            (Token::Comma, (12, 13)),
            (Token::RParen, (13, 14)),
            (Token::Colon, (15, 16)),
            (Token::Ident("Int".into()), (17, 20)),
            (Token::Equal, (21, 22)),
            (Token::Ident("a".into()), (23, 24)),
            (Token::EOF, (24, 24)),
        ]);
        let (p, f) = parse_fn(tokens);
        assert_eq!(f.airity(), 1);
        assert!(
            p.diag.is_empty(),
            "trailing comma before `)` should be accepted"
        );
    }

    #[test]
    fn fn_missing_paren_emits_diag_but_parses() {
        // def bad(a: Int: Int = a   -- missing ')'
        let tokens = toks(&[
            (Token::Def, (0, 3)),
            (Token::Ident("bad".into()), (4, 7)),
            (Token::LParen, (7, 8)),
            (Token::Ident("a".into()), (8, 9)),
            (Token::Colon, (9, 10)),
            (Token::Ident("Int".into()), (11, 14)),
            // missing RParen here
            (Token::Colon, (15, 16)),
            (Token::Ident("Int".into()), (17, 20)),
            (Token::Equal, (21, 22)),
            (Token::Ident("a".into()), (23, 24)),
            (Token::EOF, (24, 24)),
        ]);
        let (p, _f) = parse_fn(tokens);
        assert!(!p.diag.is_empty(), "should report missing `)` diagnostic");
    }

    #[test]
    fn test_if_statement_basic() {
        let tokens = toks(&[
            (Token::If, (0, 2)),
            (Token::Ident("a".into()), (3, 4)),
            (Token::Then, (5, 9)),
            (Token::Ident("xy".into()), (10, 12)),
            (Token::Else, (13, 17)),
            (Token::Ident("xy".into()), (18, 20)),
        ]);
        let (p, ifstm) = parse_expr_from_tokens(tokens);
        assert!(
            matches!(ifstm, ast::Expression::IfStm(_)),
            "expression parsed is an if statement"
        );
        if let ast::Expression::IfStm(ifstm) = ifstm {
            assert!(matches!(
                ifstm.then_expr().as_ref(),
                ast::Expression::Identifier(_)
            ));
        }
        assert!(p.diag.is_empty(), "No errors should be found");
    }

    #[test]
    fn test_if_statement_precedence() {
        // if a then 2 else 2 + 3 should be if a then 2 else (2 + 3)
        let tokens = toks(&[
            (Token::If, (0, 2)),
            (Token::Ident("a".into()), (3, 4)),
            (Token::Then, (5, 9)),
            (Token::Int(2), (10, 11)),
            (Token::Plus, (11, 12)),
            (Token::Int(3), (13, 14)),
            (Token::Else, (15, 19)),
            (Token::Ident("xy".into()), (20, 22)),
            (Token::EOF, (23, 23)),
        ]);
        let (p, ifstm) = parse_expr_from_tokens(tokens);
        assert!(
            matches!(ifstm, ast::Expression::IfStm(_)),
            "expression parsed is an if statement"
        );
        if let ast::Expression::IfStm(ifstm) = ifstm {
            println!("{:?}", ifstm.then_expr());
            assert!(
                matches!(ifstm.then_expr().as_ref(), ast::Expression::BinaryOp(_)),
                "Then expression should be 2+3, not just 2"
            );
        }
        assert!(p.diag.is_empty(), "No errors should be found");
    }

    #[test]
    fn test_if_stm_missing_then() {
        // if a then 2 else 2 + 3 should be if a then 2 else (2 + 3)
        let tokens = toks(&[
            (Token::If, (0, 2)),
            (Token::Ident("a".into()), (3, 4)),
            (Token::Int(2), (10, 11)),
            (Token::Plus, (11, 12)),
            (Token::Int(3), (13, 14)),
            (Token::Else, (15, 19)),
            (Token::Ident("xy".into()), (20, 22)),
            (Token::EOF, (23, 23)),
        ]);
        let (p, ifstm) = parse_expr_from_tokens(tokens);
        assert!(
            matches!(ifstm, ast::Expression::IfStm(_)),
            "expression parsed is an if statement"
        );
        if let ast::Expression::IfStm(ifstm) = ifstm {
            println!("{:?}", ifstm.then_expr());
            assert!(
                matches!(ifstm.then_expr().as_ref(), ast::Expression::BinaryOp(_)),
                "Then expression should be 2+3, not just 2"
            );
        }
        assert!(!p.diag.is_empty(), "Missing then keyword");
    }

    #[test]
    fn block_empty_ok() {
        // { }
        let tokens = toks(&[
            (Token::LBrace, (0, 1)),
            (Token::RBrace, (1, 2)),
            (Token::EOF, (2, 2)),
        ]);
        let (p, _e) = parse_expr_from_tokens(tokens);
        assert!(
            p.diag.is_empty(),
            "empty block should parse without diagnostics"
        );
    }

    #[test]
    fn block_decl_then_final_expr() {
        // { var x: Int = 1; x }
        let tokens = toks(&[
            (Token::LBrace, (0, 1)),
            (Token::Var, (1, 4)),
            (Token::Ident("x".into()), (5, 6)),
            (Token::Colon, (6, 7)),
            (Token::Ident("Int".into()), (8, 11)),
            (Token::Equal, (12, 13)),
            (Token::Int(1), (14, 15)),
            (Token::Semicolon, (15, 16)),
            (Token::Ident("x".into()), (17, 18)),
            (Token::RBrace, (18, 19)),
            (Token::EOF, (19, 19)),
        ]);
        let (p, _e) = parse_expr_from_tokens(tokens);
        assert!(
            p.diag.is_empty(),
            "decl + final expression should parse cleanly"
        );
    }

    #[test]
    fn block_expr_statements_and_final_expr() {
        // { a();; b(); c }
        let tokens = toks(&[
            (Token::LBrace, (0, 1)),
            (Token::Ident("a".into()), (1, 2)),
            (Token::LParen, (2, 3)),
            (Token::RParen, (3, 4)),
            (Token::Semicolon, (4, 5)),
            (Token::Semicolon, (5, 6)),
            (Token::Ident("b".into()), (6, 7)),
            (Token::LParen, (7, 8)),
            (Token::RParen, (8, 9)),
            (Token::Semicolon, (9, 10)),
            (Token::Ident("c".into()), (11, 12)),
            (Token::RBrace, (12, 13)),
            (Token::EOF, (13, 13)),
        ]);
        let (p, e) = parse_expr_from_tokens(tokens);
        assert!(
            matches!(e, ast::Expression::Block(_)),
            "expr parses to a block expression"
        );
        if let ast::Expression::Block(b) = e {
            assert_eq!(b.items.len(), 2);
            assert!(
                b.final_expr.is_some(),
                "the last expression with no semicolon is returned"
            );
        }
        assert!(p.diag.is_empty(), "No errors reported");
    }

    #[test]
    fn block_expr_statements_no_return() {
        // { a();; b(); c; }
        let tokens = toks(&[
            (Token::LBrace, (0, 1)),
            (Token::Ident("a".into()), (1, 2)),
            (Token::LParen, (2, 3)),
            (Token::RParen, (3, 4)),
            (Token::Semicolon, (4, 5)),
            (Token::Semicolon, (5, 6)),
            (Token::Ident("b".into()), (6, 7)),
            (Token::LParen, (7, 8)),
            (Token::RParen, (8, 9)),
            (Token::Semicolon, (9, 10)),
            (Token::Ident("c".into()), (11, 12)),
            (Token::Semicolon, (13, 14)),
            (Token::RBrace, (12, 13)),
            (Token::EOF, (14, 14)),
        ]);
        let (p, e) = parse_expr_from_tokens(tokens);
        assert!(
            matches!(e, ast::Expression::Block(_)),
            "expr parses to a block expression"
        );
        if let ast::Expression::Block(b) = e {
            assert_eq!(b.items.len(), 3);
            assert!(
                b.final_expr.is_none(),
                "Last expression is not returned due to semicolon"
            );
        }
        assert!(p.diag.is_empty(), "No errors reported");
    }
    #[test]
    fn block_missing_semicolon_between_exprs() {
        // { a() b(); }
        let tokens = toks(&[
            (Token::LBrace, (0, 1)),
            (Token::Ident("a".into()), (1, 2)),
            (Token::LParen, (2, 3)),
            (Token::RParen, (3, 4)),
            (Token::Ident("b".into()), (5, 6)),
            (Token::LParen, (6, 7)),
            (Token::RParen, (7, 8)),
            (Token::Semicolon, (8, 9)),
            (Token::RBrace, (9, 10)),
            (Token::EOF, (10, 10)),
        ]);
        let (p, _e) = parse_expr_from_tokens(tokens);
        assert!(!p.diag.is_empty(), "missing ';' between expressions");
    }

    #[test]
    fn block_missing_rbrace() {
        // { a();
        let tokens = toks(&[
            (Token::LBrace, (0, 1)),
            (Token::Ident("a".into()), (1, 2)),
            (Token::LParen, (2, 3)),
            (Token::RParen, (3, 4)),
            (Token::Semicolon, (4, 5)),
            (Token::EOF, (5, 5)),
        ]);
        let (p, _e) = parse_expr_from_tokens(tokens);
        assert!(
            !p.diag.is_empty(),
            "missing '}}' should produce a diagnostic"
        );
    }
}
