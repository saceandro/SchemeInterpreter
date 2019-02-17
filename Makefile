GHC=ghc
TARGET = hello fac io errorsPerLine firststep simpleParser
all: $(TARGET)

%: %.hs
	$(GHC) -o $@ $<
clean:
	rm $(TARGET) *.hi *.o
