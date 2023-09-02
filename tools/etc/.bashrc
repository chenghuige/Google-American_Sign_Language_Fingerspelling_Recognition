# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
	# We have color support; assume it's compliant with Ecma-48
	# (ISO/IEC-6429). (Lack of such support is extremely rare, and such
	# a case would tend to support setf rather than setaf.)
	color_prompt=yes
    else
	color_prompt=
    fi
fi

# some more ls aliases 
alias mbidu='sudo mount //cq01-forum-rstree01.cq01.baidu.com/img /home/users/chenghuige -o username=root,password=run,uid=1000,gid=10'
alias vpn='/opt/cisco/anyconnect/bin/vpnui'
alias chenghuige='cd /home/users/chenghuige/'
alias sublime=/opt/sublime_text/sublime_text 
alias netbeans=/home/gezi/netbeans-8.1/bin/netbeans
alias tensorboard=/usr/local/bin/tensorboard
alias ll='ls -l'
alias la='ls -A'
alias l='ls -CF'
alias lt='ls -l -t | more'
alias s='cd ~'
alias sr='cd /home/users/chenghuige/'
alias r='cd ~/rsc/app/search/sep/anti-spam/'
#alias d='cd $HOME/Desktop'
alias g='grep'
alias f='fg'
alias ..='cd ..'
alias d='~/tools/bin/d'
alias d2='~/toos/bin/d2'
alias t='~/tools/bin/t'
alias t2='~/tools/bin/t2'
alias to='~/tools/bin/to'
alias to-utf8='~/tools/bin/to-utf8'
alias c='cd'
alias v='vi'
alias rs='rsync.py'
alias e='exit'
alias h2cc='h2cc.py -a '
alias comake2='~/.jumbo/comake/comake2'
alias cms='~/.jumbo/comake/comake2 -S'
alias cm='~/.jumbo/comake/comake2'
alias crelease='cmake -DCMAKE_BUILD_TYPE=Release .'
alias creleasedebug='cmake -DCMAKE_BUILD_TYPE=ReleaseDebug .'
alias cdebug='cmake -DCMAKE_BUILD_TYPE=Debug .'
alias make='make -j4'
alias mk='/home/gezi/tools/bin/make.sh'
alias m='/home/gezi/tools/bin/make.sh'
alias mf='makef.sh'
alias mc='make clean'
alias urate='cd ~/urate/app/search/sep/anti-spam/tieba-urate'
alias root='cd  ~/sc/app/search/sep/anti-spam/'
alias gezi='cd  ~/rsc/app/search/sep/anti-spam/gezi/'
alias gezir='cd  /home/users/chenghuige/rsc/app/search/sep/anti-spam/gezi/'
alias rsc='cd ~/rsc/app/search/sep/anti-spam/rsc/'
alias rscr='cd /home/users/chenghuige/rsc/app/search/sep/anti-spam/rsc/'
alias free='chmod 777 -R *'
alias ne=netbeans
alias upad=ulipad
alias pa='print-path.py'
alias wpath='win-path.py'
alias dpath='d-path.py'
alias spath='scp-path.py'
alias rpath='remote-path.py'
alias sql='mysql -h 10.81.12.147 -uroot -proot'
alias svn=/home/gezi/.jumbo/bin/svn 
alias svn-ci='svn ci -m "ISSUE=$1" '
alias sc=svn-ci
alias svc=svn-ci 
alias svu='svn update'
alias ev='~/tools/evaluate.py'
alias evp='~/tools/evaluate.py --precision --thre '
alias gh=gen-header.sh
alias vw='~/other/vowpal_wabbit/vowpalwabbit/vw'
alias melt='cd ~/rsc/app/search/sep/anti-spam/melt/'
alias meltr='cd /home/users/chenghuige/rsc/app/search/sep/anti-spam/melt/'
alias melt2='cd ~/rsc/app/search/sep/anti-spam/melt-train'
alias melt2r='cd /home/users/chenghuige/rsc/app/search/sep/anti-spam/melt-train'
alias gp='~/rsc/app/search/sep/anti-spam/melt/tools/gbdt_predict'
alias predict='~/rsc/app/search/sep/anti-spam/melt/tools/predict'
alias tar='/bin/tar'
ulimit -c 0
#alias gcc4='export PATH=/home/users/chenghuige/.jumbo/opt/gcc-4.8.2/bin/:$PATH'
alias gccori='export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games'
alias gccv2='export PATH=/sbin:/usr/bin:/bin/:$PATH'
alias gcc4='export PATH=/opt/compiler/gcc-4.8.2/bin/:$PATH'
alias gcc49='export PATH=/home/users/chenghuige/.jumbo/opt/gcc49/bin/:$PATH'
alias gcc46='export PATH=/home/users/chenghuige/.jumbo/opt/gcc46/bin/:$PATH'
alias gcc4-default='export PATH=/opt/compiler/gcc-4.8.2/bin/:$PATH'
alias gcc48='export PATH=/home/users/chenghuige/.jumbo/opt/gcc48/bin/:$PATH'
alias gcc482='export PATH=/home/users/chenghuige/.jumbo/opt/gcc-4.8.2/bin:$PATH'
alias gcc3='export PATH=/home/users/chenghuige/.jumbo/bin:/home/users/chenghuige/hpc/client/hpc_client/bin:/home/users/chenghuige/tools/scripts:/home/users/chenghuige/work/com/tools/ubgen:/home/users/chenghuige/tools:/home/users/chenghuige/tools/bin:/home/users/chenghuige/tools/scripts/:/home/tools/tools/svn/1.6.5/64/bin/:/home/tools/bin/64/:/home/tools/tools/scmtools/usr/bin/:/home/users/chenghuige/tools:/home/tools/tools/svn/1.6.5/64/bin/:/home/tools/bin/64/:/home/tools/tools/scmtools/usr/bin/:/usr/kerberos/bin:/usr/local/bin:/bin:/usr/bin:/usr/X11R6/bin:/usr/share/baidu/bin:/home/users/chenghuige/bin'
#export PYTHONPATH=/home/users/chenghuige/python:$PYTHONPATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games
#export PATH=/home/users/chenghuige/tools:/home/users/chenghuige/tools/bin:$PATH
export PATH=~/bin:~/tools:~/tools/bin:$PATH
export PATH=/usr/local/cuda-7.0/bin:$PATH  

LD_LIBRARY_PATH=''
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH 

PYTHONPATH=''
export PYTHONPATH=~/python:$PYTHONPATH
alias gdb='~/.jumbo/bin/gdb'
#alias clear='~/.jumbo/bin/clear'

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
fi
#[[ -s "/home/gezi/.jumbo/etc/bashrc" ]] && source "/home/gezi/.jumbo/etc/bashrc"

#export LD_LIBRARY_PATH=/home/gezi/.jumbo/lib:$LD_LIBRARY_PATH 
export LD_LIBRARY_PATH=~/python-cplusplus-lib/:$LD_LIBRARY_PATH 
#export CPLUS_INCLUDE_PATH=/home/gezi/.jumbo/include/
export CUDA_HOME=/usr/local/cuda


if [ "$color_prompt" = yes ]; then
    #PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    #PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
    PS1='${debian_chroot:+($debian_chroot)}\u:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    #PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u: \w\a\]$PS1"
    #PS1='\u:\w\$ '
    ;;
*)
    ;;
esac


# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi
