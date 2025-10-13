\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mipt-thesis}[2016/06/18 Class for the MS thesis in the MIPT]

%%%%%% Class options %%%%%
\DeclareOption*{
    \ClassWarning{myclass}{Unknown option '\CurrentOption'}
}
\ProcessOptions\relax

\LoadClass[a4paper,12pt,oneside]{book}
\linespread{1.5}
%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Useful packages %%%%%
% Math packages, should be loaded before fontenc and babel for Cyrillic
\RequirePackage{amsmath,amsthm,amssymb}
\RequirePackage{mathtext}
% Fonts and lang
\RequirePackage[T1,TS1,T2A]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[english,russian]{babel}
% Formatting
\RequirePackage{geometry}
\RequirePackage{setspace}
\RequirePackage{indentfirst}
\RequirePackage{fancyhdr}
\RequirePackage{nopageno}
\RequirePackage{tocloft}
% Style file with additional packages
\RequirePackage{mipt-thesis}
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Set up correct formatting %%%%%
\geometry{left=3cm}
\geometry{right=2cm}
\geometry{top=2cm}
\geometry{bottom=2cm}
\onehalfspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Some dark TeX magic %%%%%
\addto\captionsrussian{\renewcommand{\contentsname}{Содержание}}
\addto\captionsrussian{\renewcommand{\listfigurename}{Список иллюстраций}}
\addto\captionsrussian{\renewcommand{\listtablename}{Список таблиц}}
\addto\captionsrussian{\renewcommand{\bibname}{Литература}}
\addto\captionsrussian{\renewcommand{\indexname}{Index}}
\addto\captionsrussian{\renewcommand{\figurename}{Рис.{}}}
\addto\captionsrussian{\renewcommand{\tablename}{Таблица}}
\addto\captionsrussian{\renewcommand{\partname}{Часть}}
\addto\captionsrussian{\renewcommand{\chaptername}{Глава}}
\addto\captionsrussian{\renewcommand{\appendixname}{Приложение}}
%%% Has no effect since babel package switches tags (see https://tex.stackexchange.com/questions/153182/strange-behaviour-on-redefining-contentsname, /usr/share/texlive/texmf-dist/tex/generic/babel-russian/russianb.ldf (Debian))%%%
%\renewcommand\contentsname{Содержание}
%\renewcommand\listfigurename{Список иллюстраций}
%\renewcommand\listtablename{Список таблиц}
%\renewcommand\bibname{Литература}
%\renewcommand\indexname{Index}
%\renewcommand\figurename{Рис.{}}
%\renewcommand\tablename{Таблица}
%\renewcommand\partname{Часть}
%\renewcommand\chaptername{Глава}
%\renewcommand\appendixname{Приложение}


\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \partmark{#1}%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >-2\relax
       \huge\bfseries\boldmath \partname~\thepart
       \par
       \vskip 20\p@
     \fi
     \Huge \bfseries\boldmath #2\par}%
    \@endpart}
\def\@spart#1{%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \Huge \bfseries\boldmath #1\par}%
    \@endpart}

\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \if@mainmatter
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#2}%
                       \else
                         \addcontentsline{toc}{chapter}{#2}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{#2}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}


\def\@makechapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        \huge\bfseries\boldmath \@chapapp\space \thechapter
        \par\nobreak
        \vskip 20\p@
      \fi
    \fi
    \interlinepenalty\@M
    \Huge \bfseries\boldmath #1\par\nobreak
    \vskip 40\p@
  }}
\def\@makeschapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \Huge \bfseries\boldmath  #1\par\nobreak
    \vskip 40\p@
  }}


\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                             {\normalfont\Large\raggedright}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                             {\normalfont\Large\sffamily\itshape\raggedright}}
%%%%%%%% Captions %%%%%%%%%%%%%%%%%%%%%%%%%%
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#1. #2}%
  \ifdim \wd\@tempboxa >\hsize
    #1. #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

%%%%%%%% Headings
\fancypagestyle{miptthesis}{%
    \fancyhf{}
    \fancyhead[RO, LE]{\thepage}
    \fancyhead[CO]{\leftmark}
    \fancyhead[CE]{\rightmark}
}
\pagestyle{miptthesis}
\renewcommand{\chaptermark}[1]{\markboth{\hfill\normalfont #1\hfill}{}}
\renewcommand{\sectionmark}[1]{\markright{\hfill\large\itshape \S\thesection\quad #1\hfill}}
\let\partmark=\chaptermark

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Right TOC numbering %%%%%
\renewcommand{\cftchapaftersnum}{.}
\renewcommand{\cftsecaftersnum}{.}
\renewcommand{\cftsubsecaftersnum}{.}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Easier references %%%%%
\newcommand{\formref}[1]{(\ref{#1})}
\newcommand{\picref}[1]{рис.~\ref{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Titlepage and contents %%%%%
% Set some "variables"
\renewcommand{\author}[1]{\def\mtauthortext{#1}}
\renewcommand{\title}[1]{\def\mttitletext{#1}}
\newcommand{\groupnum}[1]{\def\mtgroupnumtext{#1}}
\newcommand{\supervisor}[1]{\def\mtsupervisortext{#1}}
\newcommand{\referee}[1]{\def\mtrefereetext{#1}}

\newcommand{\addemptypage}{%
    \clearpage \mbox{} \clearpage
}

\renewcommand{\titlepage}{%
    \pagestyle{empty}
    \begin{center}
        {\bf Министерство образования и науки Российской Федерации \\
        Московский физико-технический институт (национальный исследовательский университет) \\ }
        \vspace{1cm}

        Физтех-школа радиотехники и компьютенрых технологий \\
        Кафедра мультимедийных технологий и телекоммуникаций \\

        \vspace{3em}

        Выпускная квалификационная работа магистра по специальности 03.04.01
        <<Прикладные математика и физика>>
    \end{center}

    \begin{center}
        \vspace{\fill}
        \singlespacing

        \LARGE \mttitletext{}

        \vspace{\fill}
    \end{center}


    \begin{flushright}
        Студент \mtgroupnumtext{} группы \\
        \mtauthortext{} \\
        \vspace{3em}
        Научный руководитель \\
        \mtsupervisortext{} \\
        \vspace{3em}
        Рецензент \\
        \mtrefereetext{} \\
    \end{flushright}

    \vspace{6em}

    \begin{center}
        Долгопрудный \\
        \the\year{}
    \end{center}

}

\newcommand{\abstract}{
\chapter{Аннотация}
В данной работе проводится анализ алгоритмов адаптивной компенсации нелинейных паразитных искажений, возникающих в приемо-передающих устройствах телекоммуникаций. Нелинейные искажения образуются в передатчиках базовых станций, а также мобильных устройств при прохождении сигнала через нелинейные цепи передающего тракта, усилитель мощности и дуплексер. 

Вследствие компактности расположения компонент в RF-чипсете мобильного устройства, паразитная помеха распространяется из передатчика в приёмную часть уменьшая чувствительность приёмника.

Нелинейные искажения значительно влияют на качество радиочастотных сигналов, создавая помехи в канале связи. Такие нелинейные искажения приводят к увеличению битовой ошибки (Bit Error Rate - BER) на приемнике, генерируют внутриполосные и внеполосные помеховые сигналы и ухудшают качество передачи сигналов соседних полос. Для предотвращения искажений в современных базовых станциях и сотовых устройствах широко используются методы цифровой предискажения DPD, основанные на методе адаптивной компенсации нелинейных искажений.

В работе рассматриваются задачи адаптивной нелинейной компенсации искажений в задаче цифрового предыскажения сигнала передатчика, а адаптивной компенсации паразитных помех в приёмном тракте мобильного терминала системы связи стандарта 4G/5G. Приводится математическое описание модели нелинейной помехи, а также используемых в исследовании алгоритмов.

Проводится исследование методов адаптивной компенсации нелинейных искажений, а также различных моделей нелинейных искажений радиосигналов.

Исследование алгоритмов адаптации, а также моделей нелинейных искажений, проводится на основе сигналов, сформированных на физической платформе. Сигнал помехи формируется относительно различных случаев выходной мощности усилителя.

Адаптивный корректор строится на основе классических нелинейных моделей, качественно описывающей процессы формирования нелинейных искажений, а также на основе нейросетевого подхода, позволяющего увеличить обобщающую способность адаптивной нелинейной модели. Особое внимание уделяется вопросам ускорения сходимости алгоритмов адаптации параметров нелинейной модели, а также методам сокращения ресурсов при аппаратной реализации модуля адаптивной коррекции.
}

\newcommand{\introduction}{
\chapter{Введение}
По мере совершенствования телекоммуникационных технологий возникают задачи повышения скорости передачи информации. Так например разработка систем стандарта 5G предполагает решение задач связанных с увеличением скорости передачи информации по сравнению с системами стандарта 4G. Увеличение скорости передачи требует повышения порядков цифровой модуляции. А это в свою очередь требует улучшения характеристик чувствительности приёмника.

Кроме того, важнейшим направлением развития телекоммуникационных технологий является унификация производства аппаратуры. Так основная тенденция это упрощение аналоговых трактов аппаратуры с компенсацией возникающих искажений адаптивными цифровыми корректорами \cite{dig_front_end}. 

Современным направлением решения перечисленных задач является использование адаптивных систем нелинейной коррекции, которые позволяют существенно повысить коэффициент полезного действия передающих трактов системы при сохранении характеристик линейности системы в соотвествии с принятыми международными стандартами связи \cite{3gpp_36_804}. С точки зрения пользователя мобильного терминала это означает увеличение времени автономной работы телефона. 

В данной работе рассматриваются алгоритмы подавления помех в приёмном тракте мобильного устройства, возникающих в передающем тракте путём прохождения через нелинейные элементы, такие как дуплексеры и усилители мощности \cite{dpd_thesis}. Ввиду интегрального исполнения аналоговых компонент, их характеристики имеют существенно нелинейный вид. Кроме того, ввиду компактного расположения компонент в RF-чипсете мобильного терминала, нелинейные компоненты, возникающие в передающем тракте, попадают в приемный тракт мобильного телефона, проходя по различным путям распространения, уменьшая чувствительность приёмника, а соответственно и пропускную способность нисходящего канала связи.

Адаптивная компенсация паразитных нелинейных помех позволяет улучшить пропускную способность нисходящего канала связи, повысить КПД усилителя передатчика, а также ослабить требования, предъявляемые к аналоговым дуплексерам.

В работе приводятся результаты симуляции работы модуля адаптивной коррекции для сигналов помехи различной ширины полосы, а также для различных каналов распространения помехи.
}

\newcommand{\conclusion}{
\chapter{Заключение}
%В рамках данной работы испытаны алгоритмы адаптации модели Гаммерштейна -- модели нелинейных помех. Проведён сравнительный анализ эффективности градиентных методов первого порядка, таких как блочный градиентный спуск и стохастические алгоритмы SGD-SGD, SGD-DCD. Доказана высокая скорость сходимости стохастических методов SGD-SGD и SGD-DCD к опорному уровню целевой функции NMSE, который формируется методами второго порядка. 
%
%Подавление паразитной помехи стохастическим методом SGD-SGD отличается от опорных значений компенсации, сформированных демпфированным методом Ньютона меньше, чем на 1 дБ для всех рассмотренных случаев формирования паразитной помехи. Кроме того, данный метод обладает наибольшей скоростью сходимости среди рассмотренных градиентных методов.
%
%Компенсация паразитной нелинейной помехи в приёмнике методом SGD-DCD отличается от опорных значений подавления, полученных в результате работы демпфированного метода Ньютона, меньше, чем на 4 дБ также для всех рассмотренных случаев формирования помехи.
%
%В случае работы блочного метода градиентного спуска подавление помехи отличается от опорных значений на 16 дБ и менее. Данный метод обладает худшей скоростью сходимости среди рассмотренных методов первого порядка.
%
%Предложен метод экономии ресурсов при аппаратной реализации адаптивных КИХ-фильтров. Алгоритм позволяет сократить площадь, используемую для расположения компонент КИХ-фильтра на кристалле, а также потребляемую мощность за счёт замены части умножителей сумматорами и сдвиговыми регистрами.
%
%Предложенный алгоритм позволяет сократить число умножителей несимметричного КИХ-фильтра в два раза при уровне отклонения частотной характеристики от исходной -70 дБ. При этом в случае сокращения числа умножителей симметричного КИХ-фильтра в 5 раз, уровень отклонения частотной характеристики от исходной не хуже -60 дБ.
}

\newcommand{\titlecontents}{
    %\titlepage 
    \abstract
    \newpage
    \tableofcontents 
    \introduction
}

\newcommand{\concludecontents}{
	\conclusion
	\newpage
	%\pagestyle{miptthesis}
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\endinput

% vim: syntax=tex:
