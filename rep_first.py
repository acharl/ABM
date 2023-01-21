import mesa
import random
import matplotlib.pyplot as plt
import statistics
import numpy as np 

# This ABM instance should ideally reflect the behaviour of the matching algorithm 
# as well as the reputation system. 
# 
# - Potentially we would have to simplify the matching algo a bit, while we can implement the rep system 1:1
# 
# - I suppose the most interesting aspect is to see whether there is some kind of dominance of a handful of processors which emerges 


class MarketPlace(mesa.Agent):

    avg_reward = 0
    total_rewards = 0
    total_jobs_matched = 0

    total_jobs_fulfilled = 0
    total_jobs_failed = 0 
    

    jobs_to_be_matched = [] 
    advertisements = {} 

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
    def register_job(self, job):
        if (not self.match_job(job)):
            self.jobs_to_be_matched.append(job)
        
    def register_advertisement(self, advertisement):
        self.advertisements[advertisement["processor_id"]] = advertisement

    def match_job(self, job): 

        consumer = self.model.consumers[job["consumer_id"]]

        matches = []
        def has_capacity(advertisement):
            return advertisement["capacity"] > 1 

        def is_within_budget(advertisement):
            return job["reward"] > advertisement["price_per_cpu_second"] * job["cpu_seconds"] 

        def has_min_reputation(advertisement):
            matched_processor = self.model.processors[advertisement["processor_id"]]
            return matched_processor.reputation > job['min_reputation']
        
        ads_with_capacity = [ad for ad in list(self.advertisements.values()) if has_capacity(ad)]# TODO the matchin also has to include the pricing? 

        ads_within_budget = [ad for ad in ads_with_capacity if is_within_budget(ad)]

        if job['min_reputation']:
            matches = [ad for ad in ads_within_budget if has_min_reputation(ad)]
        else: 
            matches = ads_within_budget
        
        if len(matches) > 0: 
            slots = job['slots']
            # sorted_matches = sorted(matches, key=lambda m: m['price_per_cpu_second'])[:slots]

            candidates = [self.model.processors[m["processor_id"]] for m in matches]
            sorted_candidates = sorted(candidates, key=lambda c: c.reputation)[:slots]

            if (len(sorted_candidates) == slots):
                for candidate in sorted_candidates: 
                    candidate.process_job(job, self.avg_reward, slots)
                    self.advertisements.pop(candidate.unique_id)
                
                consumer.on_process_job()
                self.total_jobs_matched += 1
                self.total_rewards += job['reward']
                self.avg_reward = self.total_rewards / self.total_jobs_matched 

                return True
        
            else: 
                return False
            

            # if job['slots'] > 1:  
            #     pass
            # else: 
            #     # TODO instead of matching the cheapest, match that of the processor with the highest reputation
            #     candidates = [self.model.processors[m["processor_id"]] for m in matches]
            #     matched_advertisement = min(matches, key=lambda x:x['price_per_cpu_second'])


            #     # matched_processor = self.model.processors[matched_advertisement["processor_id"]]
            #     matched_processor = max(candidates, key=lambda x:x.reputation)
            #     # match_ad_index = next((index for (index, d) in enumerate(matches) if d["a"] == 1), None)

            #     # remove the matched advertisement from the open advertisements
                
            #     self.advertisements.pop(matched_processor.unique_id)
            #     matched_processor.process_job(job, self.avg_reward)

            # consumer.on_process_job()
            # self.total_jobs_matched += 1
            # self.total_rewards += job['reward']
            # self.avg_reward = self.total_rewards / self.total_jobs_matched 
        
            # return True
            
        else: 
            return False
        
    def step(self):
        for job in self.jobs_to_be_matched: 
            self.match_job(job)

class ConsumerAgent(mesa.Agent):
    has_open_jobs = False
    job = {}
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.cpu_seconds = random.randint(1, 100)
        self.reward = random.randint(1, 10) * self.cpu_seconds

    def on_process_job(self):
        self.has_open_jobs = False

    def step(self):
        if self.has_open_jobs: 
            pass
        else: 
            job = {}
            min_reputation = random.uniform(0.7, 1)
            slots = random.randint(1, 10)
            has_multiple_slots = random.uniform(0, 1) > 0.5 
            job = { 
                "cpu_seconds": self.cpu_seconds, 
                "reward": self.reward * slots if has_multiple_slots else self.reward, 
                "min_reputation": min_reputation if random.uniform(0, 1) > 0.5 else None,
                "slots": slots if has_multiple_slots else 1, # TODO in presence of slots, the reward should be proportionally higher
                "consumer_id": self.unique_id
                }
            self.model.market_place.register_job(job) 
            self.has_open_jobs = True
        

# perhaps we could later have multiple processors implementing different strategies regarding price making etc. 
class ProcessorAgent(mesa.Agent):
    has_open_advertisement = False
    income = 0 
    
    advertisement = {}
    r = 0 
    s = 0 
    reputation = 0

    # type ∈ [low, medium, high] depending on success_rate
    def __init__(self, success_rate, type, unique_id, model):
        super().__init__(unique_id, model)
        self.price_per_cpu_second = random.randint(1, 100)
        self.capacity = random.randint(10, 100)
        self.success_rate = success_rate
        self.type = type

    def update_reputation(self, job, avg_reward, is_fulfilled):
        w = job['reward'] / (job['reward'] + avg_reward)
        λ = self.model.lmbda
        threshold = (self.r - self.r*λ + self.s*λ)/(self.s+1)
        if is_fulfilled:      
            self.model.market_place.total_jobs_fulfilled += 1

            if w > threshold: 
                self.r *= λ 
                self.s *= λ 
                self.r += w
        else: 
            self.r *= λ 
            self.s *= λ    
            self.s += w   
            self.model.market_place.total_jobs_failed += 1

        self.reputation = self.calculate_reputation()

    def process_job(self, job, avg_reward, slots = None):
        is_fulfilled = random.uniform(0, 1) < self.success_rate
        if is_fulfilled: 
            if slots: 
                self.income += job['reward']/slots
            else: 
                self.income += job['reward']
            self.update_reputation(job, avg_reward, True)
        else: 
            self.update_reputation(job, avg_reward, False)



    def calculate_reputation(self):
        return (self.r+1) / (self.r+self.s+2)
         
    def get_income(self): 
        return self.income

    def step(self):
        def register(): 
            self.model.market_place.register_advertisement(self.advertisement)

        # if we do have an open advertisement, then adjust pricing? 
        if self.has_open_advertisement:
            if self.advertisement["price_per_cpu_second"] > 1: 
                self.advertisement["price_per_cpu_second"] -= 1
                register()

        if not self.has_open_advertisement:
            self.advertisement = {"price_per_cpu_second": self.price_per_cpu_second, "capacity": self.capacity, "processor_id": self.unique_id}
            self.has_open_advertisement = True
            register()
        
def compute_gini(processor_wealths): 
    x = sorted(processor_wealths)
    N = len(processor_wealths)
    B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
    return 1 + (1 / N) - 2 * B

class ReputationModel(mesa.Model):
    success_rates = [0.9, 0.95, 0.99]
    types = ['low', 'medium', 'high']

    def __init__(self, M, N):

        self.num_processors = M
        self.num_consumers = N

        self.lmbda = 0.98

        self.schedule = mesa.time.BaseScheduler(self)

        self.consumers = {}
        self.processors = {}

        # introduce three kinds of processors with different success_rates 
        for i in range(len(self.success_rates)): 
            for j in range(int(M/len(self.success_rates))): 
                 unique_id = "P_" + self.types[i] + '_' + str(j)
                 a = ProcessorAgent(self.success_rates[i], self.types[i], unique_id, self)
                 self.schedule.add(a)
                 self.processors[unique_id] = a
      
        for i in range(self.num_consumers):
            unique_id = "C_"+str(i)
            a = ConsumerAgent("C_"+str(i), self)
            self.consumers[unique_id] = a
            self.schedule.add(a)

        self.market_place = MarketPlace(1, self)
        self.schedule.add(self.market_place)


    def step(self):
        self.schedule.step()

##################################################################################################

incomes_by_type = {}
reputations_by_type = {}

types = ['low', 'medium', 'high']

failure_rates = []

for i in range(10): 
    model = ReputationModel(300, 100) # TODO 
    for j in range(100):
        model.step()

    for type in model.types: 
        processors_by_type = [p if p.type == type else None for p in model.processors.values()]
        incomes = [p.get_income() for p in processors_by_type if p is not None]
        reputations = [p.reputation for p in processors_by_type if p is not None]
        if type in incomes_by_type: 
            incomes_by_type[type] += (incomes)
        else: 
            incomes_by_type[type] = incomes

        if type in reputations_by_type: 
            reputations_by_type[type] += (reputations)
        else: 
            reputations_by_type[type] = reputations

    failure_rate = model.market_place.total_jobs_failed/(model.market_place.total_jobs_failed + model.market_place.total_jobs_fulfilled)
    failure_rates.append(failure_rate)

colors = ['blue','red','green']


fig = plt.figure(figsize = (10, 5))


for i, type in enumerate(types): 
    incomes = incomes_by_type[type]

    reputations = reputations_by_type[type]
    print('############### ' + type + ' ###############')
    print('\n')
    print('MAX INC ' + str(max(incomes)))
    print('MIN INC ' + str(min(incomes)))
    print('AVG INC ' + str(sum(incomes)/len(incomes)))
    print('STDev INC ' + str(statistics.stdev(incomes)))

    print('--------------------------')
    print('MAX REP ' + str(max(reputations)))
    print('MIN REP ' + str(min(reputations)))
    print('AVG REP ' + str(sum(reputations)/len(reputations)))
    print('STDev REP ' + str(statistics.stdev(reputations)))

    print('\n')

    plt.scatter(incomes, reputations, c = colors[i], s=1)

print(failure_rates)
print('AVG failure rate', str(sum(failure_rates)/len(failure_rates)))
plt.savefig('ABM_plot.pdf')

# plt.plot(range(0, len(ginis)), ginis)
# plt.savefig('ABM_plot')
# gini = compute_gini(all_incomes)
# print(gini)
# print('avg Gini', sum(ginis)/len(ginis))
# plt.xlabel("Reward Income")
# plt.ylabel("Number of Processors")
# plt.title("Students enrolled in different courses")
# plt.hist(all_incomes, bins=range(max(all_incomes) + 1))
# plt.savefig('ABM_plot')




#❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️
# KEY Learnings
# - the impact of the reputation system depends entirely on how the matching is implemented
# 
# There are obviously many ways to implement the matching.

# It seems like there are two main approaches: 
# 
# 1)    - consider all ads which fulfill min_reputation
#       - select the cheapest
#       - in this scenario the consumer may pay less than he's willing to pay
# 
# 2)    - find all ads which are within the budget of the job
#       - select the one with the highest reputation
#       - since we're not maximizing for price, the consumer should be expected to pay more on average 
#❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️
