# Minimal Makefile
# Usage:
#   make        -> build from main.cpp
#   make clean  -> remove build artifacts
#   make nuke   -> DANGEROUS: delete everything except main.cpp (see below)

# Detect Windows vs. Unix for exe name and rm command
ifeq ($(OS),Windows_NT)
  EXE := a.exe
  RM  := del /Q
  SHELL := cmd
else
  EXE := a.out
  RM  := rm -f
endif

CXX := g++
CXXFLAGS := -O2 -std=c++17
SRC := main.cpp

all: $(EXE)

$(EXE): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXE)

clean:
	-$(RM) $(EXE)
	-$(RM) *.o
	-$(RM) *.obj
	-$(RM) *.ppm
	-$(RM) *.png
	-$(RM) *.tsv
	-$(RM) *.exe 2>NUL || true

# --- DANGEROUS ---
# Deletes everything in the current directory EXCEPT main.cpp and this Makefile.
# Use at your own risk.
nuke:
ifeq ($(OS),Windows_NT)
	@for %f in (*) do @if /I not "%f"=="main.cpp" if /I not "%f"=="Makefile" del /Q "%f"
else
	@find . -maxdepth 1 -type f ! -name 'main.cpp' ! -name 'Makefile' -exec rm -f {} +
endif

.PHONY: all clean nuke
