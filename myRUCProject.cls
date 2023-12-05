\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{myRUCProject}[Anakin's personalized project class]
% ========================= Tools for package ================================

\RequirePackage{ifthen}     % Guess what it does, I dare you
\RequirePackage{etoolbox}   % Includes a lot of useful macros for customizations
\RequirePackage{xkeyval}    % More useful macros
\RequirePackage{xparse}     % More useful macros

\RequirePackage{iftex}      % Allows for if Lua LaTeX commands


% ========================= Class tags

\global\let\tagRuc@typeProject@\iftrue\relax
    

\newif\iftagBibble@ 
\newif\iftagGlossel@ 

\newif\iftagFontFuckery@ 
\newif\iftagFinal@ 


% ========================= User Options


\DeclareOption{printBibble}{% enables bibliography via boolean flag
    \tagBibble@true
    }
    
\DeclareOption{printGloss}{% enables glossary via boolean flag
    \tagGlossel@true
    }

    
% ========================= Class Basis


\DeclareOption{finalHandin}{%
    \tagFinal@true
    }
    

\DeclareOption{a4}{%
    \PassOptionsToClass{a4paper}{report}
    \PassOptionsToPackage{a4paper,centering,vscale = 0.80,hscale = 0.75}{geometry}
    }
    
% \DeclareOption{11pt}{%
%     \PassOptionsToClass{11pt}{report}
%     }

\DeclareOption{noindent}{%
    \AtBeginDocument{%
        \setlength{\parindent}{0em} % Paragraph indents
        }
    }

\DeclareOption{doubleColumn}{%
    %\RequirePackage{multicol} % allows for merging multiple columns
    \PassOptionsToClass{twocolumn}{report}
    }



% ========================= no more Options

\DeclareOption*{%
    \ClassWarning{Project}{There isn't any `\CurrentOption' option dumbass}%
    }

% Execute `default' options
\ExecuteOptions{a4}
% Process given options
\ProcessOptions\relax

\LoadClass[11pt]{report}

% ========================= Text Packages 




% It's 2022. UTF-8 input for TeX. 
\iftutex
    \RequirePackage[no-math]{fontspec}
    \setsansfont{CMU Sans Serif}%{Arial}
    \setmainfont{CMU Serif}%{Times New Roman}
    \setmonofont{CMU Typewriter Text}%{Consolas}
    \defaultfontfeatures{Ligatures={TeX}}
    %\setmainfont{TeX Gyre Termes}
    %\RequirePackage{unicode-math}
    %\setmathfont{Stix Two Math}
    \else
        \RequirePackage[utf8]{inputenc}
        %
        % \RequirePackage{psnfss} % Font support?
        \RequirePackage{mathpazo}
        \RequirePackage{crimson}        
        % \RequirePackage{mathptmx} % Times new Roman      
        %
        \RequirePackage[T1]{fontenc} % Best imported after fonts. or something
        %
        \RequirePackage[final]{microtype}
    \fi


% ========================= Colors/Visuals/Figures Function

\RequirePackage[table, dvipsnames]{xcolor} % allows for more exciting colors
\RequirePackage{graphicx}       % supplement for the built-in graphics package
\RequirePackage{adjustbox}      % additional supplement for box adjustments
\RequirePackage{float}          % Allows for adjustments of floats
    

\RequirePackage{listings}   % For source code with syntax highlighting
\RequirePackage{verbatim}   % For more text enviroment options


\RequirePackage[labelfont={bf}, hypcap=false]{caption} % For figure coloration
\RequirePackage{subcaption} % Subfigures


\RequirePackage[outline]{contour} % Glow around text

% ========================= Format margins, headers and footers

\RequirePackage{datetime} % for generating dates and time
% \RequirePackage{lastpage} % Pretty much exclusively used for page labeling

\RequirePackage{enumitem}    % for list formatting

\RequirePackage{indentfirst} % Corrects formatting inconsistancy
\RequirePackage{setspace} % Line spacing
% \RequirePackage{ragged2e} % Text Justifier

% ======================== Class Bibliography Options =====================

\PassOptionsToPackage{%
    sorting     = none,%
    maxbibnames = 99,%
    url         = false,%
    style       = numeric-comp,%
    }{biblatex}
\PassOptionsToPackage{%
    nonumberlist,%
    toc, %
    xindy, %
    style   = index, %
    }{glossaries}
    
\AtEndOfClass{
    \glsdisablehyper
    \setacronymstyle{long-sc-short}
    }


% ======================== Custom Packages ================================

\RequirePackage{myMathPreamble}
\RequirePackage{myReferencePreamble}

% ====================== Font config



% \let\oldnormalfont\normalfont
% \def\normalfont{\oldnormalfont\mdseries}

% Removes the serif for default font
% \renewcommand{\familydefault}{\sfdefault}

% \renewcommand{\normalsize}{\fontsize{10pt}{12pt}\selectfont}

% \DeclareMathSizes{10}{18}{12}{8}   % For size 10 text
% \DeclareMathSizes{11}{19}{13}{9}   % For size 11 text
% \DeclareMathSizes{12}{20}{14}{10}  % For size 12 text




% ======== Microtype
\microtypesetup{%
    activate={true,nocompatibility},%
    tracking=true,%
    kerning=true,%
    spacing=true,%
    factor=1100,%
    stretch=10,%
    shrink=10}
% activate={true,nocompatibility} - activate protrusion and expansion
% final - enable microtype; use "draft" to disable
% tracking=true, kerning=true, spacing=true - activate these techniques
% factor=1100 - add 10% to the protrusion amount (default is 1000)
% stretch=10, shrink=10 - reduce stretchability/shrinkability (default is 20/20)

\microtypecontext{spacing=nonfrench}

\SetProtrusion{encoding={*}, family={*}, series={*}, size={6,7}}{%
              1={ ,750},%
              2={ ,500},%
              3={ ,500},%
              4={ ,500},%
              5={ ,500},%
              6={ ,500},%
              7={ ,600},%
              8={ ,500},%
              9={ ,500},%
              0={ ,500}}

\SetExtraKerning[unit=space] % Needs kerning = true
    {encoding={*}, family={*}, series={*}, size={scriptsize,footnotesize,small,normalsize}}{%
    \textendash={400,400}, % en-dash, add more space around it
    "28={ ,150}, % left bracket, add space from right
    "29={150, }, % right bracket, add space from left
    \textquotedblleft={ ,150}, % left quotation mark, space from right
    \textquotedblright={150, }, % right quotation mark, space from left
    \textquoteleft={ ,150}, % left quotation mark, space from right
    \textquoteright={150, }, % right quotation mark, space from left
    } %
    
\SetExtraKerning[unit=space] % For titles
   {encoding={*}, family={qhv}, series={b}, size={large,Large}}
   {1={-200,-200}, \textendash={400,400}}

\SetTracking{encoding={*}, shape=sc}{40} % Fucks with small caps


% ====================== Tweaks and settings ================================

\AtBeginEnvironment{table}{%
    \rowcolors{1}{white}{gray!30} % Table row coloration
    \setlength{\arrayrulewidth}{1pt} % Table line widths
    \setlength{\tabcolsep}{5pt} % Table padding horizontally
    \renewcommand{\arraystretch}{1.5} % Table padding vertically
    \sisetup{per-mode = symbol}
    }

\newcommand{\red}{\color{red!80!black}}
\newcommand{\blue}{\color{blue!80!black}}


%\renewcommand{\thefootnote}{\arabic{footnote}}



\renewcommand{\footnoterule}{\vfill\kern -3pt \hrule width \textwidth \kern 2.6pt}
    
% \justifying

\tolerance=9999         % Gives more whitespace
\emergencystretch=10pt  % Gives more whitespace
\hyphenpenalty=10000    % Removes hyphanation
\exhyphenpenalty=100    % Allows for existing hyphans to stay



% ===================== Page Styles

\RequirePackage{geometry} % Page margins
\RequirePackage{fancyhdr} % Allows for easy fancy page layouts
\RequirePackage{titlesec} % Allows for easy editing of of headers




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
\fancyhf{} % Resetting formatting
\fancyhead[L]{\nouppercase{\leftmark}}
\fancyhead[R]{Modeling Project}


% \if
% \fancyfoot[C]{$\left[\thepage\text{ of }\pageref{LastPage}\right]$}
\fancyfoot[C]{$\left[\thepage\right]$}


\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}


\fancypagestyle{plain}{\fancyhead{}\renewcommand{\headrule}{}}

% \fancypagestyle{toc}{%
%     \fancyhf{}%
%     \renewcommand{\headrulewidth}{0.0pt}
%     \renewcommand{\footrulewidth}{0.0pt}
%     \fancyhead[R]{\thepage}%
%     \fancyfoot[C]{$\left[\thepage\right]$}
%     }

\fancyhfoffset{0pt}


% ======================== Custom Headers ================================

% \titleformat{<command>}[<shape>]{<format>}{<label>}{<sep>}{<before-code>}[<after-code>]
% \titlespacing{<command>}{<left>}{<before-sep>}{<after-sep>}

\titleformat{\chapter} % command
    [block] % shape
    {\bfseries\huge\usefont{T1}{cmr}{bx}{n}\selectfont} % format
    {\thechapter} % label
    {0.5ex} % sep
    {\bfseries\fontsize{14}{12} \huge\centering} % before-code
    [\vspace{-1.5ex}\rule{\textwidth}{0.5pt}\\ \vspace{-2ex}\rule{\textwidth}{0.5pt}] % after-code

\titlespacing*{\chapter}%
    {0ex}%
    {-3.55em}%
    {1em}%


\titleformat{\section}% command
    [hang] % shape
    {\bfseries\Large\usefont{T1}{qhv}{b}{n}\selectfont} % format
    {\thesection} % label
    {0.3em} % sep
    {\bfseries\fontsize{14}{12} \selectfont} % before-code
    [\vskip 0.5ex \titlerule] % after-code

\titlespacing*{\section}
    {0pt}%
    {*4}%
    {*4}%


\titleformat{\subsection}
    [hang]
    {\bfseries\large\usefont{T1}{qhv}{b}{n}\selectfont}
    {\thesubsection\hspace{1em}\textemdash}
    {1em}
    {}
    

% ========================================================================

\RequirePackage{import}





\endinput