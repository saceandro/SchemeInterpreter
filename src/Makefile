GHC=ghc
TARGET = simpleParser
all: $(TARGET)

%: %.hs
	$(GHC) -o $@ $<
clean:
	rm $(TARGET) *.hi *.o
