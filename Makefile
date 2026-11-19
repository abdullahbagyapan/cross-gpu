# SPDX-License-Identifier: MIT
PYTHON ?= python3
PREFIX ?= /usr/local
DESTDIR ?=
DATADIR ?= $(PREFIX)/share
MANDIR ?= $(DATADIR)/man/man1
BASHCOMPDIR ?= $(DATADIR)/bash-completion/completions
ZSHCOMPDIR ?= $(DATADIR)/zsh/site-functions
FISHCOMPDIR ?= $(DATADIR)/fish/vendor_completions.d

.PHONY: test check install clean

test:
	$(PYTHON) -m unittest discover -s tests -v

check:
	$(PYTHON) -m compileall -q cross_gpu tests
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m cross_gpu.embed_binary verify

install:
	$(PYTHON) -m pip install --no-build-isolation --no-deps \
		--prefix="$(PREFIX)" $(if $(DESTDIR),--root="$(DESTDIR)",) .
	install -d "$(DESTDIR)$(MANDIR)"
	install -m 644 docs/cross-gpu.1 "$(DESTDIR)$(MANDIR)/cross-gpu.1"
	install -d "$(DESTDIR)$(BASHCOMPDIR)"
	install -m 644 completions/cross-gpu.bash "$(DESTDIR)$(BASHCOMPDIR)/cross-gpu"
	install -d "$(DESTDIR)$(ZSHCOMPDIR)"
	install -m 644 completions/_cross-gpu "$(DESTDIR)$(ZSHCOMPDIR)/_cross-gpu"
	install -d "$(DESTDIR)$(FISHCOMPDIR)"
	install -m 644 completions/cross-gpu.fish "$(DESTDIR)$(FISHCOMPDIR)/cross-gpu.fish"

clean:
	rm -rf build dist *.egg-info
	find cross_gpu tests -name '__pycache__' -type d -prune -exec rm -rf {} +
	find cross_gpu tests -name '*.py[co]' -delete
