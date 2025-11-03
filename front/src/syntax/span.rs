use std::ops::Range;

use crate::source::FileId;

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

pub type FileSpan = CtxSpan<FileId>;

impl FileSpan {
    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn file(&self) -> &FileId {
        &self.ctx
    }
}

impl From<(logos::Span, FileId)> for FileSpan {
    fn from((s, f): (logos::Span, FileId)) -> Self {
        Self {
            start: s.start,
            end: s.end,
            ctx: f,
        }
    }
}
