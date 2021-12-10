#!/bin/bash

qpdf 0_preamble.pdf --pages 0_preamble.pdf 2-z -- section1.pdf
qpdf 0_paper.pdf --pages 0_paper.pdf 2-z -- section2.pdf
qpdf 5_appendixA.pdf --pages 5_appendixA.pdf 2-z -- appendixA.pdf
qpdf 6_appendixB.pdf --pages 6_appendixB.pdf 2-z -- appendixB.pdf
qpdf 7_appendixC.pdf --pages 7_appendixC.pdf 2-z -- appendixC.pdf


pdfunite 0_titlepage.pdf section1.pdf section2.pdf appendixA.pdf appendixB.pdf appendixC.pdf dissertation.pdf
