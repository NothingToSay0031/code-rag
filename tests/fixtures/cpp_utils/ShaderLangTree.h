#pragma once

#include <string>
#include <vector>

namespace engine
{
    struct ShaderLangTreeNode
    {
        std::string name;
        std::vector<ShaderLangTreeNode> children;
    };

    class ShaderLangTree
    {
    public:
        void AddNode(const std::string& name);
        const ShaderLangTreeNode* FindNode(const std::string& name) const;

    private:
        ShaderLangTreeNode m_root;
    };
}
