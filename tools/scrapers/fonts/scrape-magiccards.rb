#sm_url ='http://magiccards.info/sitemap.html#en'
sets = ['ori', 'm15', 'm14', 'm13', 'm12',
        'm11', '10e', '9e', '8e', '7e', '6e',
        'soi', 'ogw', 'bfz', 'dtk', 'frf',
        'ktk', 'jou', 'bng', 'ths', 'dgm', 'gtc', 'rtr',
        'avr', 'dka', 'isd', 'roe', 'wwk', 'zen', 'arb', 
        'cfx', 'ala', 'eve' 'shm', 'mt', 'lw', 
        'fut',  'pc', 'ts', 'tsts', 'cs', 'ai', 'ia',
        'di', 'gp', 'rav', 'sok', 'bok', 'chk', '5dn', 'ds',
        'mi', 'sc', 'le', 'on', 'ju', 'tr', 'od', 'ap',
        'ps', 'in', 'pr', 'ne', 'mm', 'ud', 'ul',
        'us', 'ex', 'sh', 'tp', 'wl', 'vi', 'mr',
        'hl', 'fe', 'dk', 'lg', 'aq', 'an',
        '5e', '4e', 'rv', 'un', 'be', 'al', 'vma',
        'me3', 'me2', 'med'
        ]

sets.each do |set|
  url = "http://magiccards.info/scans/en/#{set}"
  1.upto(500) do |i|
    print("Getting set #{set} number #{i}")
    puts `curl #{url}/#{i}.jpg -o #{set}-#{i}.jpg`
  end
end


