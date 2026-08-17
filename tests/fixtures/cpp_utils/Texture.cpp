#include "Texture.h"

namespace engine
{
    Texture::Texture()
        : m_width(0)
        , m_height(0)
    {
    }

    void Texture::Resize(int width, int height)
    {
        m_width = width;
        m_height = height;
    }

    int Texture::GetWidth() const
    {
        return m_width;
    }
}
