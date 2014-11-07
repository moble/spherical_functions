=begin
  Jekyll tag to include Markdown text from _includes directory preprocessing with Liquid <https://gist.github.com/mignev/7759676>.
  Usage:
    {% markdown <filename> %}
=end
module Jekyll
  class MarkdownTag < Liquid::Tag
    def initialize(tag_name, text, tokens)
      super
      @text = text.strip
    end

    def render(context)
      tmpl = File.read File.join Dir.pwd, @text
      site = context.registers[:site]
      converter = site.getConverterImpl(Jekyll::Converters::Markdown)
      tmpl = (Liquid::Template.parse tmpl).render site.site_payload
      html = converter.convert(tmpl)
    end
  end
end
Liquid::Template.register_tag('markdown', Jekyll::MarkdownTag)
