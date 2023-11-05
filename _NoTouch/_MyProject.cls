\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{MyProject}[Anakin's personalized Project writing class]

% ========================= Class Basis


\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}

% Report options
\DeclareOption{10pt}{
  \PassOptionsToClass{\CurrentOption}{report}
}
\DeclareOption{11pt}{
  \PassOptionsToClass{\CurrentOption}{report}
}
\DeclareOption{a4paper}{
  \PassOptionsToClass{\CurrentOption}{report}
}

\DeclareOption*{%
\ClassWarning{RUC Project}{Unknown option `\CurrentOption'}%
}

\ExecuteOptions{a4paper, 11pt}

\ProcessOptions\relax

\LoadClassWithOptions{report}
 
% ========================= Preamble Packages ================================

\RequirePackage{ifthen}
\RequirePackage{etoolbox} % Includes a lot of useful macros for customizations
%\usepackage{iftex} % Allows for if Lua LaTeX commands

% ========================= Colors/Visuals

\RequirePackage{graphicx}
\RequirePackage[table, dvipsnames]{xcolor}
    \rowcolors{1}{white}{gray!30} % Table row coloration
\RequirePackage{blindtext}

% ========================= Format page header and footer

\RequirePackage{datetime}
\RequirePackage{verbatim}
\RequirePackage{lastpage}
\RequirePackage{fancyhdr}
    \pagestyle{fancy}
    \fancyhf{}
    \fancyhead[LO]{\leftmark}
    %\fancyhead[RO]{Semester 4 Project}
    \fancyfoot[CO]{[\thepage]}
    \renewcommand{\headrulewidth}{2pt}
    \renewcommand{\footrulewidth}{1pt}
    \renewcommand{\footnoterule}{\vfill\kern -3pt \hrule width \textwidth \kern 2.6pt}
    \fancyhfoffset{0pt}
    
% =========================  Margin and paragraph formatting

\RequirePackage{indentfirst} % Indents first paragraph
\RequirePackage{enumitem}
\RequirePackage{setspace} % Line spacing
\RequirePackage[margin=2cm]{geometry} % Page margins
%\doublespacing
    \setlength{\headheight}{15pt}
    \setlength{\marginparwidth}{2cm}
    \setlength{\parindent}{2em} % Paragraph indents
    \setlength{\parskip}{2ex} % Paragraph seperation
    \setlength{\arrayrulewidth}{0pt} % Table line widths
    \setlength{\tabcolsep}{3pt} % Table padding horizontally
    \renewcommand{\arraystretch}{1.5} % Table padding vertically
%\RequirePackage{lipsum} % Produces paragraphs of text
\RequirePackage{ragged2e} % Text Justifier
    \justifying


\tolerance=9999 % Gives more whitespace
\emergencystretch=10pt % Gives more whitespace
\hyphenpenalty=10000 % Removes hyphanation
\exhyphenpenalty=100 % Allows for existing hyphans to stay

% =========================  Margin and paragraph formatting

\RequirePackage{indentfirst} % Indents first paragraph
\RequirePackage{enumitem}
\RequirePackage{setspace} % Line spacing
\RequirePackage[margin=2cm]{geometry} % Page margins
%\doublespacing
    \setlength{\headheight}{15pt}
    \setlength{\marginparwidth}{2cm}
    \setlength{\parindent}{2em} % Paragraph indents
    \setlength{\parskip}{2ex} % Paragraph seperation
    \setlength{\arrayrulewidth}{0pt} % Table line widths
    \setlength{\tabcolsep}{3pt} % Table padding horizontally
    \renewcommand{\arraystretch}{1.5} % Table padding vertically

%\RequirePackage{lipsum} % Produces paragraphs of text
\RequirePackage{ragged2e} % Text Justifier
    \justifying


\tolerance=9999 % Gives more whitespace
\emergencystretch=10pt % Gives more whitespace
\hyphenpenalty=10000 % Removes hyphanation
\exhyphenpenalty=100 % Allows for existing hyphans to stay

% ====================== Margin and Notes ================================

\RequirePackage{marginnote}
\RequirePackage{todonotes}

% ============== Reference Functions, Figure Functions =========================

\RequirePackage[labelfont={bf}]{caption} % For figure coloration
    \def\red{\color{red!80!black}}
    \def\blue{\color{blue!80!black}}
    \DeclareCaptionFont{red}{\red}
    \DeclareCaptionFont{blue}{\blue}
    \DeclareCaptionFont{black}{\color{black}}
    \captionsetup[figure]{labelfont=red}
    %\captionsetup[subfigure]{labelfont=black}
    %
    %\captionsetup[table]{labelfont=blue}
    %\captionsetup[subtable]{labelfont=black}
    %
\RequirePackage{multicol} % allows for merging multiple columns
\RequirePackage{adjustbox} % allows you to modify figures
\RequirePackage{subcaption} % Subfigures
\RequirePackage{enumitem} % for list formatting
    \interlinepenalty=10000
\RequirePackage{array} % used for easy access in working memory
\RequirePackage[outline]{contour} % Glow around text
    \contourlength{1.4pt}


%\RequirePackage[sorting=none,
%                %style=nature,
%                %citestyle=nature,
%                maxbibnames=99,
%                url=false,
%                ]{biblatex} % For better function

% ======================== Custom Headers ================================

\RequirePackage{titlesec} % Allows for easy editing of of headers
% \titleformat{<command>}[<shape>]{<format>}{<label>}{<sep>}{<before-code>}[<after-code>]
% \titlespacing{<command>}{<left>}{<before-sep>}{<after-sep>}

\titleformat{\chapter} % command
    [block] % shape
    {\bfseries\huge} % format
    {\thechapter} % label
    {0.5ex} % sep
    {\bfseries\fontsize{14}{12} \huge\centering} % before-code
    [\vspace{-1.5ex}\rule{\textwidth}{0.5pt}\\ \vspace{-2ex}\rule{\textwidth}{0.5pt}] % after-code

\titlespacing*{\chapter}%
    {0ex}%
    {-3.55em}%
    {1em}%


\titleformat{\section}% command
    [block] % shape
    {\bfseries\Large} % format
    {\thesection} % label
    {0.3em} % sep
    {\bfseries\fontsize{14}{12} \selectfont} % before-code
    [\vskip 0.5ex \titlerule] % after-code

\titlespacing*{\section}
    {0pt}%
    {*4}%
    {*4}%


\titleformat{\subsection}
    [block]
    {\bfseries\large}
    {\thesubsection\hspace{1em}-}
    {1em}
    {}
    




%\AtBeginDocument{}


\AtEndDocument{
    \printbibliography[title={References}]
    \addcontentsline{toc}{chapter}{References}
    %\printindex
    \printglossaries
    \addcontentsline{toc}{chapter}{Glossary}
}


% ========================================================================
\endinput