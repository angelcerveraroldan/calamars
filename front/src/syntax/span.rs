use std::ops::Range;

use calamars_core::ids;

#[derive(Debug, Clone, PartialEq)]
pub struct CtxSpan<Ctx> {
    pub start: usize,
    pub end: usize,
    pub ctx: Ctx,
}

impl<Ctx> CtxSpan<Ctx> {
    pub fn into_range(&self) -> Range<usize> {
        self.start..self.end
    }
}

impl<Ctx: Copy> Copy for CtxSpan<Ctx> {}

pub type Span = CtxSpan<()>;

impl From<Range<usize>> for Span {
    fn from(value: Range<usize>) -> Self {
        Self {
            start: value.start,
            end: value.end,
            ctx: (),
        }
    }
}

impl Span {
    pub fn dummy() -> Self {
        Span::from(0..0)
    }
}

pub type FileSpan = CtxSpan<ids::FileId>;

impl FileSpan {
    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn file(&self) -> &ids::FileId {
        &self.ctx
    }
}

impl From<(logos::Span, ids::FileId)> for FileSpan {
    fn from((s, f): (logos::Span, ids::FileId)) -> Self {
        Self {
            start: s.start,
            end: s.end,
            ctx: f,
        }
    }
}
