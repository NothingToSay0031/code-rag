#include "FileManager.h"

#include <cstdio>

namespace engine
{
    bool FileManager::FileExists(const char* path) const
    {
        FILE* fp = fopen(path, "rb");
        if (fp == nullptr)
        {
            return false;
        }
        fclose(fp);
        return true;
    }

    int FileManager::GetFileSize(const char* path) const
    {
        return 0;
    }
}
