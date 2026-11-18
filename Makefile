PYTHON ?= python3
PREFIX ?= /usr/local
DESTDIR ?=
MANDIR ?= $(PREFIX)/share/man/man1

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

clean:
	rm -rf build dist *.egg-info
	find cross_gpu tests -name '__pycache__' -type d -prune -exec rm -rf {} +
	find cross_gpu tests -name '*.py[co]' -delete
