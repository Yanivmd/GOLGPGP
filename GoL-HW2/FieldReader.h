#include <string>

class FieldReader;
class PatternBlock;

///////////////////////////////////////////////////////////////////////////////

typedef unsigned char byte;

enum CellState {
    DEAD  = 0,
    ALIVE = 1
};

///////////////////////////////////////////////////////////////////////////////

// Conway's Game of Life field reader class.
/*!
  The class contains functions for parsing Game-Of-Life fields in Life 1.05 format.
  The format is desribed in http://conwaylife.com/wiki/Life_1.05
  The parse() and readFile() functions load pattern-blocks from the file, and
  the buildField() function generates a window of the field with a given size.
*/
class FieldReader {
public:
    //! Constructors, destructor and assignment operator
    FieldReader();
    FieldReader(const FieldReader&);
    ~FieldReader();
    FieldReader& operator=(const FieldReader& rhs);

    //! Parses a string in Life 1.05 format.
    /*!
        Generates a list of Pattern Blocks in 'patternBlocks'
        Returns false on failure
    */
    bool parse(const std::string& text);
    
    //! Reads a file in Life 1.05 format and parses it
    /*!
        Returns false on failure
    */
    bool readFile(const std::string& filename);             // read game field from file
    
    //! Generates a field of the field
    /*!
        The field is of specified size, and has coordinate (0,0) in the middle.
        I.e., the most upper-left cell in the field is coordinate (-sizeX/2, -sizeY/2)
        
        The function should be called after loading the array of Pattern Blocks 
        using parse() or readFile().
        
        Live cells outside the field margins are not included.

        The function writes the result into 'field' provided by the user. It is assumed
        that 'field' is of size at least (sizeX*sizeY).
    */
    bool buildField(byte* field, int sizeX, int sizeY);

    static bool writeField(const std::string& filename, byte* field, int sizeX, int sizeY);
    
    PatternBlock* patternBlocks;
    int countPatternBlocks;
};

class PatternBlock {
public:
    PatternBlock();
    PatternBlock(const PatternBlock&);
    ~PatternBlock();
    PatternBlock& operator=(const PatternBlock &rhs);

    // parse text to load values. Returns false on error.
    bool parse(const std::string& text);

    static const size_t MAX_LINE_LENGTH = 80;
    
	// upper left conner coords
	int origX;
	int origY;

    // 2D pattern field
    byte *pfield;

    // pattern field dimensions
	int maxX;   //< limited by MAX_LINE_LENGTH
	int maxY;  
};