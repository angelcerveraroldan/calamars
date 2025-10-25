use crate::syntax::{span::Span, token::Token};
use chumsky::{
    error::Rich,
    extra::{self, Full},
    input::{MapExtra, ValueInput},
    span::SimpleSpan,
};
use proptest::option;

pub trait TokenInput<'a>: ValueInput<'a, Token = Token, Span = SimpleSpan> {}
impl<'a, I> TokenInput<'a> for I where I: ValueInput<'a, Token = Token, Span = SimpleSpan> {}

pub type ParserErr<'a> = extra::Err<Rich<'a, Token>>;

/// An identifier
#[derive(Debug, Clone, PartialEq)]
pub struct Ident {
    /// The identifier
    ident: String,
    /// Where it is found in the source
    span: Span,
}

impl Ident {
    pub fn new(ident: String, span: Span) -> Self {
        Self { ident, span }
    }

    pub fn ident(&self) -> &str {
        &self.ident
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

/// A Calamars module / file
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub imports: Vec<ClImport>,
    pub items: Vec<ClDeclaration>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClImport {
    idents: Vec<Ident>,
    total_span: Span,
}

impl ClImport {
    pub fn new(idents: Vec<Ident>, total_span: Span) -> Self {
        Self { idents, total_span }
    }
}

/// Any one thing in the Cl language
#[derive(Debug, Clone, PartialEq)]
pub enum ClItem {
    Declaration(ClDeclaration),
    Expression(ClExpression),

    // TODO:
    Import(ClImport),
}

impl ClItem {
    pub fn span(&self) -> Span {
        match self {
            ClItem::Declaration(cl_declaration) => cl_declaration.span(),
            ClItem::Expression(cl_expression) => cl_expression.span(),
            ClItem::Import(_) => todo!("import not yet handled"),
        }
    }
}

/// Calamars Base Type kind and value
#[derive(Debug, Clone, PartialEq)]
pub enum ClLiteralKind {
    Integer(i64),
    Real(f64),
    String(String),
    Boolean(bool),
    Char(char),
    Array(Vec<ClLiteral>),
}

/// Calamars Base Type, along with its span in the source
#[derive(Debug, Clone, PartialEq)]
pub struct ClLiteral {
    kind: ClLiteralKind,
    span: Span,
}

impl ClLiteral {
    pub fn new(kind: ClLiteralKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn kind(&self) -> &ClLiteralKind {
        &self.kind
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

impl From<ClLiteral> for ClExpression {
    fn from(value: ClLiteral) -> Self {
        ClExpression::Literal(value)
    }
}

/// Types for Calamars
#[derive(Debug, Clone, PartialEq)]
pub enum ClType {
    /// Basic / standard types such as Int, String, Char, Real, ...
    /// as well as types that require many segments, such as people.Person
    Path { segments: Vec<Ident>, span: Span },
    /// An array of some type such as [Int]
    Array { elem_type: Box<Self>, span: Span },
    /// A function (I1, I2, I3, ...) -> (O1, O2, O3, ...)
    Func {
        inputs: Vec<Option<Self>>,
        output: Box<Option<Self>>,
        /// Lambdas have a very clear "full span", where as functions dont
        span: Option<Span>,
    },
}

impl ClType {
    pub fn span(&self) -> Option<Span> {
        match self {
            ClType::Path { span, .. }
            | ClType::Array { span, .. }
            | ClType::Func {
                span: Some(span), ..
            } => Some(*span),
            _ => None,
        }
    }
}

/// All types of possible expressions
#[derive(Debug, Clone, PartialEq)]
pub enum ClExpression {
    Literal(ClLiteral),
    Identifier(Ident),

    UnaryOp(ClUnaryOp),
    BinaryOp(ClBinaryOp),

    IfStm(IfStm),
    FunctionCall(FuncCall),

    Block(ClCompoundExpression),
}

impl ClExpression {
    pub fn span(&self) -> Span {
        match self {
            ClExpression::Literal(cl_literal) => cl_literal.span(),
            ClExpression::Identifier(ident) => ident.span(),
            ClExpression::UnaryOp(cl_unary_op) => cl_unary_op.span(),
            ClExpression::BinaryOp(cl_binary_op) => cl_binary_op.span(),
            ClExpression::IfStm(if_stm) => if_stm.span(),
            ClExpression::FunctionCall(func_call) => func_call.span(),
            ClExpression::Block(cl_compound_expression) => cl_compound_expression.total_span(),
        }
    }

    /// If something is retuend, span to the expression being returned
    ///
    /// If the expression has Unit return type, then return nothing
    ///
    /// TODO: If statements are not handled well due to having two arms. This will get worse with
    /// match statements that have `n` arms.
    pub fn returning_span(&self) -> Option<Span> {
        if let ClExpression::Block(cl_compound_expression) = self {
            cl_compound_expression.return_span()
        } else {
            Some(self.span())
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Neg,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add,      // +
    Sub,      // -
    Times,    // *
    Pow,      // ^
    Div,      // /
    Concat,   // ++
    Geq,      // >=
    Leq,      // <=
    EqEq,     // ==
    NotEqual, // !=

    Or,  // or
    Xor, // xor
    And, // and
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClUnaryOp {
    operator: UnaryOperator,
    on: Box<ClExpression>,
    span: Span,
}

impl From<ClUnaryOp> for ClExpression {
    fn from(value: ClUnaryOp) -> Self {
        ClExpression::UnaryOp(value)
    }
}

impl ClUnaryOp {
    pub fn new(operator: UnaryOperator, on: Box<ClExpression>, span: Span) -> Self {
        Self { operator, on, span }
    }

    pub fn operator(&self) -> &UnaryOperator {
        &self.operator
    }

    pub fn inner_exp(&self) -> &Box<ClExpression> {
        &self.on
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClBinaryOp {
    operator: BinaryOperator,
    left: Box<ClExpression>,
    right: Box<ClExpression>,
    span: Span,
}

impl From<ClBinaryOp> for ClExpression {
    fn from(value: ClBinaryOp) -> Self {
        ClExpression::BinaryOp(value)
    }
}

impl ClBinaryOp {
    pub fn new(
        operator: BinaryOperator,
        left: Box<ClExpression>,
        right: Box<ClExpression>,
        span: Span,
    ) -> Self {
        Self {
            operator,
            left,
            right,
            span,
        }
    }

    pub fn operator(&self) -> &BinaryOperator {
        &self.operator
    }

    pub fn lhs(&self) -> &Box<ClExpression> {
        &self.left
    }

    pub fn rhs(&self) -> &Box<ClExpression> {
        &self.right
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfStm {
    predicate: Box<ClExpression>,
    then: Box<ClExpression>,
    otherwise: Box<ClExpression>,

    span: Span,
}

impl IfStm {
    pub fn new(
        predicate: Box<ClExpression>,
        then: Box<ClExpression>,
        otherwise: Box<ClExpression>,
        span: Span,
    ) -> Self {
        Self {
            predicate,
            then,
            otherwise,
            span,
        }
    }

    pub fn span(&self) -> SimpleSpan {
        self.span
    }

    pub fn pred_span(&self) -> SimpleSpan {
        self.predicate.span()
    }

    pub fn then_span(&self) -> SimpleSpan {
        self.then.span()
    }

    pub fn else_span(&self) -> SimpleSpan {
        self.otherwise.span()
    }

    pub fn then_expr(&self) -> &Box<ClExpression> {
        &self.then
    }

    pub fn else_expr(&self) -> &Box<ClExpression> {
        &self.otherwise
    }

    pub fn pred(&self) -> &Box<ClExpression> {
        &self.predicate
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncCall {
    func_name: Ident,
    params: Vec<ClExpression>,
    span: Span,
}

impl FuncCall {
    pub fn new(func_name: Ident, params: Vec<ClExpression>, span: Span) -> Self {
        Self {
            func_name,
            params,
            span,
        }
    }

    pub fn params(&self) -> &Vec<ClExpression> {
        &self.params
    }

    pub fn name(&self) -> &str {
        &self.func_name.ident
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn name_span(&self) -> Span {
        self.func_name.span()
    }
}

/// An expression in the form
///
/// {
///    (<cl_item>)*
///    (<cl_expression>)?
/// }
#[derive(Debug, Clone, PartialEq)]
pub struct ClCompoundExpression {
    pub items: Vec<ClItem>,
    pub final_expr: Option<Box<ClExpression>>,

    span: Span,
}

impl ClCompoundExpression {
    pub fn new(items: Vec<ClItem>, final_expr: Option<Box<ClExpression>>, span: Span) -> Self {
        Self {
            items,
            final_expr,
            span,
        }
    }

    pub fn total_span(&self) -> SimpleSpan {
        self.span
    }

    pub fn return_span(&self) -> Option<SimpleSpan> {
        self.final_expr.clone().map(|x| x.span())
    }
}

// DECLARATIONS

#[derive(Debug, Clone, PartialEq)]
pub enum ClDeclaration {
    Binding(ClBinding),
    Function(ClFuncDec),
}

impl ClDeclaration {
    pub fn span(&self) -> Span {
        match self {
            ClDeclaration::Binding(cl_binding) => cl_binding.span(),
            ClDeclaration::Function(cl_func_dec) => cl_func_dec.span(),
        }
    }

    pub fn name_span(&self) -> Span {
        match self {
            ClDeclaration::Binding(cl_binding) => cl_binding.name_span(),
            ClDeclaration::Function(cl_func_dec) => cl_func_dec.name_span(),
        }
    }
}

/// Value and Variable declaration
#[derive(Debug, Clone, PartialEq)]
pub struct ClBinding {
    pub vname: Ident,
    pub vtype: Option<ClType>,
    pub assigned: Box<ClExpression>,
    pub mutable: bool,

    span: Span,
}

impl ClBinding {
    pub fn new(
        vname: Ident,
        vtype: Option<ClType>,
        assigned: Box<ClExpression>,
        mutable: bool,
        span: Span,
    ) -> Self {
        Self {
            vname,
            vtype,
            assigned,
            mutable,
            span,
        }
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn name_span(&self) -> Span {
        self.vname.span()
    }

    pub fn type_span(&self) -> Option<Span> {
        self.vtype.clone().map(|ty| ty.span()).flatten()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClFuncDec {
    pub doc_comment: Option<String>,

    fname: Ident,
    input_idents: Vec<Ident>,
    functype: ClType,
    body: ClExpression,
    span: Span,
}

impl ClFuncDec {
    pub fn new(
        fname: Ident,
        inputs: Vec<(Ident, Option<ClType>)>,
        out_type: Option<ClType>,
        body: ClExpression,
        span: Span,
        doc_comment: Option<String>,
    ) -> Self {
        let cap = inputs.len();
        let (input_idents, inputs) = inputs.into_iter().fold(
            (Vec::with_capacity(cap), Vec::with_capacity(cap)),
            |(mut idents, mut tys), (ident, ty)| {
                idents.push(ident);
                tys.push(ty);
                (idents, tys)
            },
        );

        Self {
            fname,
            input_idents,
            functype: ClType::Func {
                inputs,
                span: None,
                output: out_type.into(),
            },
            body,
            span,
            doc_comment,
        }
    }

    pub fn airity(&self) -> u16 {
        self.input_idents.len() as u16
    }

    pub fn fntype(&self) -> &ClType {
        &self.functype
    }

    pub fn name(&self) -> &String {
        &self.fname.ident
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn name_span(&self) -> Span {
        self.fname.span()
    }

    pub fn body(&self) -> &ClExpression {
        &self.body
    }

    pub fn input_idents(&self) -> &Vec<Ident> {
        &self.input_idents
    }

    pub fn output_span(&self) -> Option<Span> {
        match &self.functype {
            ClType::Func { output, .. } => match output.as_ref() {
                Some(a) => a.span(),
                None => None,
            },
            _ => None,
        }
    }
}
