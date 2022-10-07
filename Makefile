all:researchproposal.tex translatedHeadings.tex
	if eq($(OS),Windows_NT)
		pdflatex-dev.exe researchproposal
	else
		pdflatex researchproposal
clean:
	rm *.aux
	rm *.log
	rm *.bcf
	rm *.xml