execute pathogen#infect()
set t_Co=256
"colorscheme wombat256mod
syntax on
set nu
set autoindent
set cindent
set ruler
set nocompatible
set nobackup
if has('mouse')
	set mouse=a
endif

filetype plugin indent on

"nmap sx :x!<CR>
"nmap q :q!<CR>
nmap \v :set mouse=v<CR>
nmap \a :set mouse=a<CR>
set pastetoggle=<F5>
let Tlist_Show_One_File=1
let Tlist_Exit_OnlyWindow=1

let g:winManagerWindowLayout='FileExplorer|TagList'
nmap wm :WMToggle<cr>

let g:miniBufExplMapWindowNavArrows = 1
nnoremap <silent> <F10> :A<CR>

let g:SuperTabDefaultCompletionType="context"  
let g:SuperTabRetainCompletionType=2
let g:SuperTabDefaultCompletionType="<C-N><C-P>"

map <F9> :!/usr/bin/ctags -R --c++-kinds=+p --fields=+iaS --extra=+q .<CR>
""map <F7> :tprevious<CR>
"map <F8> :tnext<CR>
"set tags=~/rsc/tags;
"nmap n :set tags=;<CR>
"nmap t :set tags=~/rsc/tags<CR>
"au BufNewFile,BufRead *.rb set softtabstop=2 | set shiftwidth=2  
"au FileType ruby set softtabstop=2 | set shiftwidth=2 
"autocmd FileType c,cpp,py,pl set shiftwidth=4 | set expandtab
set tabstop=2
set shiftwidth=2
set softtabstop=2
set noexpandtab

set incsearch
set ignorecase
set smartcase
set hlsearch
nmap \q :nohlsearch<CR>
"
"set nocp
"set fdm=marker
"set bs=2
"
"set encoding=gbk
"set fileencoding=gbk
"set fileencodings=utf-8,gbk,ucs-bom
set clipboard+=unnamed

set showmatch
"colorscheme desert
set fenc=utf-8
set encoding=utf-8
set fileencoding=gbk
set fileencodings=gbk,utf-8,gb18030,utf-16,big5
set termencoding=gbk

set fenc=gbk
"set fenc=utf-8
"set encoding=utf-8
set fileencoding=gbk
set fileencodings=gbk,utf-8,gb18030,utf-16,big5
set termencoding=gbk

set filetype=python
au BufNewFile,BufRead *.py,*.pyw setf python
filetype plugin on  
autocmd FileType python set omnifunc=pythoncomplete#Complete
set ofu=syntaxcomplete#Complete
autocmd FileType python runtime! autoload/pythoncomplete.vim
let g:miniBufExplMapWindowNavVim = 1"Ctrl-<hjkl> to move to window 
let g:miniBufExplTabWrap = 1 " make tabs show complete (no broken on two lines)
let g:SuperTabClosePreviewOnPopupClose = 1

nnoremap <F1> :call MyCyclebuffer(0)<CR>
nnoremap <Leader>1 :call MyCyclebuffer(0)<CR>
inoremap <F1> <ESC>:call MyCyclebuffer(0)<CR>
nnoremap <F2> :call MyCyclebuffer(1)<CR>
nnoremap <Leader>2 :call MyCyclebuffer(1)<CR>
inoremap <F2> <ESC>:call MyCyclebuffer(1)<CR>
" Cycle Through buffers 
" from MiniBufExplorer, modified by DJW
function! MyCyclebuffer(forward)
" Change buffer (keeping track of before and after buffers)
let l:origBuf = bufnr('%')
if (a:forward == 1)
bn!
else
bp!
endif
let l:curBuf = bufnr('%')
" Skip any non-modifiable buffers, but don't cycle forever
" This should stop us from stopping in any of the [Explorers]
while getbufvar(l:curBuf, '&modifiable') == 0 && l:origBuf != l:curBuf
if (a:forward == 1)
bn!
else
bp!
endif
let l:curBuf = bufnr('%')
endwhile
endfunction

" <F4> delete buffer
nnoremap <F4> :bd<CR>
nnoremap <Leader>4 :bd<CR>
inoremap <F4> <ESC>:bd<CR>

" <F3> or Ctrl-S update buffer
nnoremap <C-S> :update<CR>
inoremap <C-S> <C-O>:update<CR>
vnoremap <C-S> <C-C>:update<CR>
nnoremap <F3> :update<CR>
inoremap <F3> <C-O>:update<CR>

set dictionary+=~/tools/templates/hive.txt
"inoremap ( ()<ESC>i
"inoremap [ []<ESC>i
"inoremap { {}<ESC>i
"inoremap < <><ESC>i
:nmap \e :NERDTreeToggle<CR>
:nmap \b :b<CR>
:nmap \q :q<CR>
:nmap \qa :qall<CR>
:nmap \x :x<CR>
:nmap \xa :xall<CR>
:nmap \s :update<CR>
"autocmd vimenter * NERDTree 
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTreeType") && b:NERDTreeType == "primary") | q | endif

"powerline
"set fillchars+=stl: ,stlnc:
"set nocompatible
"set t_Co=256
"let g:Powerline_cache_enabled = 1
"let g:Powerline_cache_file='~/.vim/bundle/powerline/Powerline.cache'
"set laststatus=2   " Always show the statusline"


"python from powerline.vim import setup as powerline_setup
"python powerline_setup()
"python del powerline_setup

let g:syntastic_ignore_files=[".*\.py$"]
:set autowrite
