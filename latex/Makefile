LATEX=pdflatex
BIBTEX=bibtex
TEXFILE=informe

all: $(TEXFILE).pdf

$(TEXFILE).pdf: $(TEXFILE).tex refs.bib
	$(LATEX) $(TEXFILE).tex
	$(BIBTEX) $(TEXFILE) || true
	$(LATEX) $(TEXFILE).tex
	$(LATEX) $(TEXFILE).tex

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.gz *.fls *.fdb_latexmk $(TEXFILE).pdf

.PHONY: all clean
