prog :=xnixperms

debug ?=

$(info debug is $(debug))

ifdef debug
  release :=
  target :=debug
  extension :=debug
else
  release :=--release
  target :=release
  extension :=
endif

build:
	cargo build $(release)

install:
	cp target/$(target)/$(prog) ~/bin/$(prog)-$(extension)

test:
	cargo test
	./testing/golden.py
	cargo fmt --all --check

all: build test

help:
	@echo "usage: make $(prog) [debug=1]"
