
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{MyProject}[Anakin's personalized Project writing class]

% ========================= Class Basis

\LoadClassWithOptions{report}

\DeclareOption*{%
 \PackageWarning{MyProject}{Unknown option ‘\CurrentOption’}%
 }
 \ProcessOptions\relax

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
    
% ======================== Custom Headers ================================
% Sauce:
% https://www.ctan.org/pkg/titlesec
% https://www.overleaf.com/learn/latex/Sections_and_chapters

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
    



\pagenumbering{arabic}


%\AtBeginDocument{⟨code⟩}


%\AtEndDocument{⟨code⟩}