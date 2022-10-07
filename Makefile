all:researchproposal.tex translatedHeadings.tex
	pdflatex-dev.exe researchproposal
clean:
	rm *.aux
	rm *.log
	rm *.bcf
	rm *.xml