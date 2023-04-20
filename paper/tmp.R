


\begin{rotate}{180}
\includegraphics{graphic.pdf}
\end{rotate}


\pagenumbering{gobble}

\newcommand{\Lpagenumber}{\ifdim\textwidth=\linewidth\else\bgroup
\dimendef\margin=0
\ifodd\value{page}\margin=\oddsidemargin
\else\margin=\evensidemargin
\fi
\raisebox{\dimexpr -\topmargin-\headheight-\headsep-0.03\linewidth}[0pt][0pt]{
  \rlap{\hspace{\dimexpr \margin+\textheight+\footskip}
    \llap{\rotatebox{90}{\thepage}}}}
\egroup\fi}
\AddEverypageHook{\Lpagenumber}


\begin{landscape}
\pagestyle{empty}
\newpage
