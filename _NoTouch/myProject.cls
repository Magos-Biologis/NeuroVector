\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{myProject}[Anakin's personalized project class]
% ========================= Preamble Packages ================================

% It's 2022. UTF-8 input for TeX. 
\RequirePackage[utf8]{inputenc}

%\RequirePackage{csquotes} % Babel wants it
%\RequirePackage{babel} 
\RequirePackage[nomath]{lmodern} % Latin modern font


% Best imported after fonts.
\RequirePackage[T1]{fontenc} % I believe this encodes fonts into your PDF.


% ========================= Tools for package ================================

\RequirePackage{ifthen}     % Guess what it does, I dare you
\RequirePackage{etoolbox}   % Includes a lot of useful macros for customizations
\RequirePackage{xkeyval}     % More useful macros
\RequirePackage{xparse}     % More useful macros

%\usepackage{iftex} % Allows for if Lua LaTeX commands

% ========================= Class Basis


\DeclareOption{draft}{
    \PassOptionsToClass{\CurrentOption}{report}
    \PassOptionsToPackage{\CurrentOption}{graphicx}
}

\DeclareOption{a4}{
    \PassOptionsToClass{a4paper}{report}
    \PassOptionsToPackage{a4paper}{geometry}
}

\DeclareOption{noindent}{
    \AtBeginDocument{\noindent}
}


% ========================= User Options


\DeclareOption{RaggedRight}{
    \PassOptionsToPackage{document}{ragged2e}
}

\DeclareOption{printGloss}{
    \PassOptionsToPackage{\CurrentOption}{_NoTouch/myReferencePreamble}
}
\DeclareOption{printUnit}{
    \PassOptionsToPackage{\CurrentOption}{_NoTouch/myReferencePreamble}
}
\DeclareOption{printAll}{
    \PassOptionsToPackage{print,printUnit,printMath}{_NoTouch/myReferencePreamble}
}


% ======================== Class Bibliography Options =====================

\PassOptionsToPackage{%
        sorting     = none,%
        maxbibnames = 99,%
        url         = false,%
        style       = numeric-comp,%
    }{biblatex}
\PassOptionsToPackage{%
        toc, %
        xindy, %
        style   = index, %
    }{glossaries}
    
\AtEndOfClass{
    \glsdisablehyper
    \setacronymstyle{long-sc-short}
}

% ========================= no more Options

\DeclareOption*{%
    \ClassWarning{Project}{There isn't any `\CurrentOption' option dumbass}%
}

% Execute `default' options
\ExecuteOptions{}

% Process given options
\ProcessOptions\relax

\LoadClass[a4paper, 11pt]{report}

% Colors/Visuals/Figures Functions

\RequirePackage[table, dvipsnames]{xcolor} % allows for more exciting colors
\RequirePackage{graphicx}       % supplement for the built-in graphics package
\RequirePackage{adjustbox}      % additional supplement for box adjustments
    

\RequirePackage{listings}   % For source code with syntax highlighting
\RequirePackage{verbatim}   % For more text enviroment options


\RequirePackage[labelfont={bf}]{caption} % For figure coloration
\RequirePackage{subcaption} % Subfigures
\RequirePackage{multicol} % allows for merging multiple columns

    
\RequirePackage{array} % used for easy access in working memory
\RequirePackage[outline]{contour} % Glow around text
    
% Format margins, headers and footers

\RequirePackage{datetime} % for generating dates and time
\RequirePackage{lastpage} 

\RequirePackage[%
        a4paper, centering,%
        vscale = 0.80, hscale = 0.75,%
        %total = {170mm,250mm},
    ]{geometry} % Page margins

\RequirePackage{fancyhdr} % for generating dates and time

% Paragraph formatting

\RequirePackage{enumitem} % for list formatting
\RequirePackage{indentfirst}

\RequirePackage{setspace} % Line spacing
\RequirePackage{ragged2e} % Text Justifier

% ======================== Custom Packages ================================

\RequirePackage{_NoTouch/myMathPreamble}

\RequirePackage{_NoTouch/myReferencePreamble}


% ====================== Tweaks and settings ================================

\AtBeginEnvironment{tabular}{%
    \rowcolors{1}{white}{gray!30} % Table row coloration
    \setlength{\arrayrulewidth}{0pt} % Table line widths
    \setlength{\tabcolsep}{3pt} % Table padding horizontally
    \renewcommand{\arraystretch}{1.5} % Table padding vertically
}

\providecommand{\red}{\color{red!80!black}}
\providecommand{\blue}{\color{blue!80!black}}

\DeclareCaptionFont{red}{\red}
\DeclareCaptionFont{blue}{\blue}
\DeclareCaptionFont{black}{\color{black}}
\captionsetup[figure]{labelfont=red}
\captionsetup[subfigure]{labelfont=black}
%
\captionsetup[table]{labelfont=blue}
\captionsetup[subtable]{labelfont=black}
%

\interlinepenalty=10000


\contourlength{1.4pt}


\setlength{\parindent}{2em} % Paragraph indents
\setlength{\parskip}{2ex} % Paragraph seperation
\setlength{\marginparwidth}{2cm}
\setlength{\headheight}{14pt}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[LO]{\leftmark}
\fancyhead[RO]{Modeling Project}
\fancyfoot[CO]{[\thepage]}
\renewcommand{\headrulewidth}{2pt}
\renewcommand{\footrulewidth}{1pt}
\renewcommand{\footnoterule}{\vfill\kern -3pt \hrule width \textwidth \kern 2.6pt}
\fancyhfoffset{0pt}
    
\justifying

\tolerance=9999         % Gives more whitespace
\emergencystretch=10pt  % Gives more whitespace
\hyphenpenalty=10000    % Removes hyphanation
\exhyphenpenalty=100    % Allows for existing hyphans to stay


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
    

% ========================================================================
\endinput