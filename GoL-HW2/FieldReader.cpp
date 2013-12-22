#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "FieldReader.h"

using std::string;
using std::vector;

typedef struct PatternLine_t { byte line[PatternBlock::MAX_LINE_LENGTH]; } PatternLine;

FieldReader::FieldReader() : patternBlocks(NULL), countPatternBlocks(0)
{
    ;
}

FieldReader::FieldReader(const FieldReader& orig) : patternBlocks(NULL), countPatternBlocks(0)
{
    if (orig.patternBlocks != NULL)
    {
        countPatternBlocks = orig.countPatternBlocks;
        patternBlocks = new PatternBlock[countPatternBlocks];
        if (patternBlocks == NULL)
        {
            countPatternBlocks = 0;
            return;
        }
        for(int pb = 0; pb < countPatternBlocks; pb++)
        {
            patternBlocks[pb] = PatternBlock(orig.patternBlocks[pb]);
        }
    }
}

FieldReader::~FieldReader()
{
    delete[] patternBlocks;
    patternBlocks = NULL;
}

FieldReader& FieldReader::operator=(const FieldReader& rhs)
{
    if (this != &rhs)
    {
        countPatternBlocks = 0;
        patternBlocks = NULL;
        if (rhs.patternBlocks != NULL)
        {
            countPatternBlocks = rhs.countPatternBlocks;
            patternBlocks = new PatternBlock[countPatternBlocks];
            if (patternBlocks == NULL)
            {
                countPatternBlocks = 0;
                return *this;
            }
            for(int pb = 0; pb < countPatternBlocks; pb++)
            {
                patternBlocks[pb] = PatternBlock(rhs.patternBlocks[pb]);
            }
        }
    }
    return *this;
}

bool FieldReader::readFile(const std::string& filename)
{
    std::ifstream f;
    f.open(filename.c_str());
    std::stringstream sstr;
    if (f)
    {
        sstr << f.rdbuf();
        return parse(sstr.str());
    }
    return false;
}

bool FieldReader::parse(const std::string& text)
{
    const string _HEADER = "#Life 1.05";
    const string _DESCRIPTION = "#D";
    const string _NORMAL = "#N";     //< normal rules (default); 
    const string _RULES = "#R";     //< custom rules (not supported)
    const string _PATTERN = "#P";   //< pattern block
    const char _DEAD = '.';
    const char _ALIVE = '*';

    std::stringstream ss(text);
    std::string line;

    // check header
    line = "";
    std::getline(ss,line);
	// fix \r\n line endings
	if ((line.size() > 0) && (line[line.size() - 1] == '\r'))
			line.resize(line.size() - 1);
	// compare header
	if (0 != line.compare(_HEADER))
    {
    	std::cout << "File header mismatch\n";
    	std::cout << "Expected: '" << _HEADER << "'\n";
    	std::cout << "Read:     '" << line << "'\n";
        return false;
    }
    
    vector<string> pattern_block_texts;
    bool reading_pblock = false;
    std::stringstream pblock;

    // read contents
    while (std::getline(ss,line,'\n'))
    {
        string prefix;

    	// fix \r\n line endings
    	if ((line.size() > 0) && (line[line.size() - 1] == '\r'))
    			line.resize(line.size() - 1);

        if (reading_pblock)
        {
            if (line.at(0) == _DEAD || line.at(0) == _ALIVE)
            {
                pblock << line << '\n';
                continue;
            }
            else if (line.at(0) == '#')
            {
                // pattern block description finished 
                pattern_block_texts.push_back(pblock.str());
                reading_pblock = false;
            }
        }

        if (0 == line.compare(0,_PATTERN.size(),_PATTERN))
        {
            reading_pblock = true;
            pblock.str("");     //clear
            pblock << line << '\n';
        }

        if (line.size() == 0)
            continue;

        if (line.at(0) != '#')
            return false;
    }

    if (reading_pblock)
    {
        // pattern block description finished 
        pattern_block_texts.push_back(pblock.str());
        reading_pblock = false;
    }

    patternBlocks = new PatternBlock[pattern_block_texts.size()];
    if (patternBlocks == NULL)
        return false;
    countPatternBlocks = pattern_block_texts.size();

    bool ok = true;
    for(int pb = 0; pb < (int)pattern_block_texts.size(); pb++)
    {
        ok = patternBlocks[pb].parse(pattern_block_texts[pb]);
        if (!ok) break;
    }
    if (!ok)
    {
        delete[] patternBlocks;
        countPatternBlocks = 0;
        return false;
    }
    return true;
}

bool FieldReader::buildField(byte* field, int sizeX, int sizeY)
{
    if (field == NULL)
        return false;

    if(patternBlocks == NULL)
        return false;       //< call parse() or readFile() before calling this func

    std::fill_n(field, sizeX * sizeY, DEAD);

    int left = -sizeX/2;
    int right = sizeX + left -1;
    int up = -sizeY/2;
    int down = sizeY + up -1;

    for (int i=0; i<countPatternBlocks; i++)
    {
        int bleft = patternBlocks[i].origX;           //< leftmost
        int bright = bleft + patternBlocks[i].maxX -1;//< rightmost
        int bup = patternBlocks[i].origY;             //< uppermost
        int bdown = bup + patternBlocks[i].maxY -1;   //< lowest

        if( bright < left ||
            bleft > right ||
            bdown < up    ||
            bup > down)
        {
            // pattern block totally outside of field
            continue;
        }

        if( bleft >= left   &&
            bright <= right &&
            bup >= up   &&
            bdown <= down)
        {
            // pattern block totally inside of field
            int fieldX = bleft - left;
            int fieldY = bup - up;
            byte* ptr = &field[fieldY * sizeX + fieldX];
            byte* bptr = patternBlocks[i].pfield;
            for (int row = 0; row < patternBlocks[i].maxY; ++row)
            {
                std::copy(bptr, bptr + patternBlocks[i].maxX, ptr);
                ptr += sizeX;
                bptr += patternBlocks[i].maxX;
            }
            continue;
        }

        //pattern block partially in field
        {
            byte* ptr = field;
            byte* bptr = patternBlocks[i].pfield;
            for (int row = 0; row < patternBlocks[i].maxY; ++row)
            {
                int posY = -up + bup + row;
                if (posY >= 0 && posY < sizeY)
                {
                    for (int col = 0; col < patternBlocks[i].maxX; ++col)
                    {
                        int posX =  -left + bleft + col;
                        if (posX >= 0 && posX < sizeX)
                            ptr[posY * sizeX + posX] = bptr[row * patternBlocks[i].maxX + col];
                    }
                }
            }
        }
    }

    return true;
}

bool FieldReader::writeField(const std::string& filename, byte* field, int sizeX, int sizeY)
{
    std::ofstream f;
    f.open(filename.c_str());
    if (!f)
        return false;
    
    f << "#Field-size " << sizeX << " " << sizeY << '\n';
    std::stringstream sstr;
    byte* ptr = field;
    for (int row=0; row<sizeY; ++row)
    {
        for (int col=0; col<sizeX; ++col)
        {
            sstr << (*ptr++ == DEAD ? '.' : '*');
        }
        sstr << '\n';
    }
    f << sstr.rdbuf();
    f.close();
    return true;
}

///////////////////////////////////////////////////////////////////////////////

PatternBlock::PatternBlock() :
    origX(0), origY(0), maxX(0), maxY(0), pfield(NULL)
{
    ;
}

PatternBlock::PatternBlock(const PatternBlock& orig)
{
    origX = orig.origX;
    origY = orig.origY;
    maxX = orig.maxX;
    maxY = orig.maxY;
    if (orig.pfield != NULL)
    {
        pfield = new byte[maxX * maxY];
        std::copy(orig.pfield, orig.pfield + maxX * maxY, pfield);
    }
    else 
        pfield = NULL;
}

PatternBlock::~PatternBlock()
{
    delete[] pfield;
    pfield = NULL;
}

PatternBlock& PatternBlock::operator=(const PatternBlock& rhs)
{
    if (this != &rhs)
    {
        origX = rhs.origX;
        origY = rhs.origY;
        maxX = rhs.maxX;
        maxY = rhs.maxY;
        delete[] pfield;
        pfield = NULL;
        if (rhs.pfield != NULL)
        {

            pfield = new byte[maxX * maxY];
            std::copy(rhs.pfield, rhs.pfield + maxX * maxY, pfield);
        }
    }
    return *this;
}

bool PatternBlock::parse(const std::string& text)
{
    const string _PREFIX = "#P";
    const char _DEAD = '.';
    const char _ALIVE = '*';

    // read lines
    std::stringstream ss(text);
    string line;
    
    // parse header line and check prefix
    string prefix;
    ss >> prefix;
    if (0 != prefix.compare(_PREFIX))
        return false;       //< invalid header
    // read upper left coordinates
    ss >> origX;
    ss >> origY;

    // skip header line
    if (!std::getline(ss,line,'\n'))
        return false;
    
    // read lines
    vector<PatternLine> lines;
    while(std::getline(ss,line,'\n'))
    {
    	// fix \r\n line endings
    	if ((line.size() > 0) && (line[line.size() - 1] == '\r'))
    			line.resize(line.size() - 1);

        if (line.size() == 0)
            continue;
        int pos = 0;
        PatternLine pline;
        std::fill_n(pline.line,sizeof(pline.line),DEAD);
        for(string::iterator it = line.begin(); it != line.end(); ++it) 
        {
            char c = *it;
            if (c == _ALIVE)
                pline.line[pos] = ALIVE;
            else if (c != _DEAD)
                return false;
            ++pos;
        }
        maxX = std::max(maxX, pos);
        lines.push_back(pline);
    }
    maxY = std::max(maxY, (int)lines.size());

    // allocate and fill field 
    pfield = new byte[maxX * maxY];
    byte* wptr = pfield;
    for (int row = 0; row < maxY; ++row)
    {
        std::copy(lines[row].line,lines[row].line + maxX,wptr);
        wptr += maxX;
    }

    return true;
}
