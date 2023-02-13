import simplenlg

nlgFactory = simplenlg.NLGFactory()
# subject = nlgFactory.createNounPhrase("the", "woman")
# subject.setPlural(True)
# sentence = nlgFactory.createClause(subject, "smoke")
# sentence.setFeature(simplenlg.Feature.NEGATED, False)
sentence = nlgFactory.createSentence()
sentence.addComponent(simplenlg.StringElement("the woman"))
sentence.addComponent(simplenlg.StringElement("do"))
sentence.addComponent(simplenlg.StringElement("smoke"))
realiser = simplenlg.realiser.english.Realiser()
print(realiser.realiseSentence(sentence))
